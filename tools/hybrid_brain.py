#!/usr/bin/env python3
"""
Hybrid Brain — Main memory API server with optional multi-tenant support.

The core entry point for the memory system. Provides HTTP endpoints
for hybrid search (Qdrant vector + FalkorDB graph + BM25 keyword + neural reranking)
and memory commit with A-MAC quality gating.

Set ENABLE_TENANTS=1 or pass --tenant-mode to enable per-agent isolation via
agent_id scoped Qdrant filters. In tenant mode, each agent operates in an
isolated memory namespace. Same hybrid search pipeline.

Usage:
    python3 hybrid_brain.py                    # Start server on port 7777
    python3 hybrid_brain.py --port 7778        # Custom port
    python3 hybrid_brain.py --test             # Run test queries
    python3 hybrid_brain.py --tenant-mode        # Multi-tenant mode
    ENABLE_TENANTS=1 python3 hybrid_brain.py     # Same, via env var
"""

import argparse
import json
import os
import re
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import redis
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# BM25 hybrid reranking — core pipeline component
BM25_AVAILABLE = False
try:
    from bm25_search import hybrid_rerank as bm25_rerank
    BM25_AVAILABLE = True
except ImportError:
    try:
        from tools.bm25_search import hybrid_rerank as bm25_rerank
        BM25_AVAILABLE = True
    except ImportError:
        pass
print("[HybridBrain] BM25 reranking: enabled", flush=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8006/rerank")
FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = int(os.environ.get("FALKORDB_PORT", "6380"))
GRAPH_NAME = os.environ.get("FALKORDB_GRAPH", "brain")

# Tenant mode toggle: set ENABLE_TENANTS=1 for per-agent isolation
TENANT_MODE = os.environ.get("ENABLE_TENANTS", "0") == "1"

# A/B test toggle: set DISABLE_FALKORDB=1 to skip all graph queries
FALKORDB_DISABLED = os.environ.get("DISABLE_FALKORDB", "0") == "1"
if FALKORDB_DISABLED:
    print("[HybridBrain] ⚠ FalkorDB DISABLED via DISABLE_FALKORDB env var", flush=True)

qdrant = QdrantClient(url=QDRANT_URL)

# Neural reranker — checked dynamically per-request, not cached at startup
# This avoids the race condition where reranker is still loading when hybrid_brain starts
def is_reranker_available():
    try:
        _r = requests.post(RERANKER_URL, json={"query": "ping", "passages": ["ping"]}, timeout=2)
        return _r.status_code == 200
    except Exception:
        return False

print("[HybridBrain] Neural reranker: dynamic (checked per-request, port 8006)", flush=True)

STOP_WORDS = {
    "what", "who", "when", "where", "how", "why", "is", "are", "was",
    "were", "the", "a", "an", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "about",
    "my", "our", "their", "his", "her", "its", "this", "that",
    "do", "does", "did", "has", "have", "had", "be", "been",
    "will", "would", "could", "should", "can", "may", "might",
    "not", "no", "yes", "all", "any", "some", "every", "each",
    "connected", "related", "between", "knows", "know", "tell",
    "me", "us", "you", "find", "search", "get", "show", "list",
    "i", "we", "they", "he", "she", "it",
}

def get_embedding(text, prefix="search_query: "):
    """Get embedding from local Ollama nomic-embed-text.
    Task prefixes improve quality: 'search_query: ' for queries, 'search_document: ' for docs.
    Retries on timeout/failure with graceful degradation."""
    
    prefixed_text = f"{prefix}{text}" if prefix else text
    max_retries = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": prefixed_text}, timeout=35)
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"][0]
            elif "embedding" in data:
                return data["embedding"]
            raise ValueError(f"Unexpected embedding response: {list(data.keys())}")
        except requests.exceptions.Timeout:
            print(f"[Embedding] Timeout on attempt {attempt+1}/{max_retries}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception("Embedding service timeout after retries")
        except requests.exceptions.ConnectionError as e:
            print(f"[Embedding] Connection error: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception(f"Embedding service unavailable: {e}")
        except Exception as e:
            print(f"[Embedding] Error: {e}", flush=True)
            raise

def get_embedding_safe(text, default_action="fail", prefix="search_query: "):
    """Get embedding with graceful degradation.
    
    Args:
        text: Text to embed
        default_action: "fail" (raise), "empty" (return zero vector), or "skip" (return None)
        prefix: Task prefix ("search_query: " or "search_document: ")
    Returns:
        Embedding vector or None if skipped
    """
    try:
        return get_embedding(text, prefix=prefix)
    except Exception as e:
        print(f"[Embedding] Failed ({default_action}): {e}", flush=True)
        
        if default_action == "empty":
            # Return zero vector for graceful degradation
            return [0.0] * 768  # nomic-embed output size
        elif default_action == "skip":
            return None
        else:  # fail
            raise

def neural_rerank(query, results, top_k=None):
    """Rerank results using bge-reranker-v2-m3 on GPU1.
    Falls back to original ordering if reranker is unavailable."""
    if not results:
        return results

    passages = []
    for r in results:
        text = r.get("text", "")[:1000]
        source = r.get("source", "")
        title = r.get("title", "")
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if source:
            parts.append(f"Source: {source}")
        parts.append(text)
        passages.append(" | ".join(parts))

    try:
        resp = requests.post(RERANKER_URL, json={
            "query": query,
            "passages": passages
        }, timeout=15)
        resp.raise_for_status()
        scores = resp.json().get("scores", [])

        if len(scores) != len(results):
            return results

        for i, r in enumerate(results):
            r["rerank_score"] = scores[i]

        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k] if top_k else reranked
    except Exception as e:
        print(f"[HybridBrain] Reranker error: {e}", flush=True)
        return results

def _load_known_entities():
    """Load known entities from config file. Returns (persons_set, orgs_set, projects_set)."""
    entities_path = os.environ.get("KNOWN_ENTITIES_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "known_entities.json"))
    try:
        with open(entities_path) as f:
            data = json.load(f)
        return (
            set(data.get("persons", [])),
            set(data.get("organizations", [])),
            set(data.get("projects", [])),
        )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[HybridBrain] Known entities config not found or invalid ({e}), using empty lists", flush=True)
        return set(), set(), set()

def extract_entities_fast(text):
    """Fast regex-based entity extraction for real-time commit pipeline.
    Returns list of (name, type) tuples."""
    entities = []
    seen = set()

    # Known entities loaded from config file
    KNOWN_PERSONS, KNOWN_ORGS, KNOWN_PROJECTS = _load_known_entities()

    text_lower = text.lower()

    for name in KNOWN_PERSONS:
        if name.lower() in text_lower and name not in seen:
            seen.add(name)
            entities.append((name, "Person"))

    for name in KNOWN_ORGS:
        if name.lower() in text_lower and name not in seen:
            seen.add(name)
            entities.append((name, "Organization"))

    for name in KNOWN_PROJECTS:
        if name.lower() in text_lower and name not in seen:
            seen.add(name)
            entities.append((name, "Project"))

    # Capitalized multi-word names (likely people/orgs not in known lists)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        name = match.group(1)
        if name not in seen and len(name) > 4:
            seen.add(name)
            entities.append((name, "Person"))

    return entities

def write_to_graph(point_id, text, entities, timestamp):
    """Write entities + memory node to FalkorDB graph 'brain'.
    Returns (success, connected_entities_list) where connected_entities is a list of entity names that were linked."""
    if FALKORDB_DISABLED:
        return True, []
    try:
        r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT)
        r.ping()
    except Exception as e:
        print(f"[Graph-Commit] FalkorDB connection error: {e}", flush=True)
        return False, []

    text_preview = text[:500].replace('\n', ' ')
    ts = timestamp

    # Create Memory node (parameterized query to prevent Cypher injection)
    try:
        r.execute_command('GRAPH.QUERY', GRAPH_NAME,
            "MERGE (m:Memory {id: $id}) SET m.text = $text, m.created_at = $ts",
            '--params', json.dumps({"id": str(point_id), "text": text_preview, "ts": ts}))
    except Exception as e:
        print(f"[Graph-Commit] Memory node error: {e}", flush=True)
        return False, []

    # Create Entity nodes + relationships and collect connected entity names
    connected_entities = []
    for name, etype in entities:
        try:
            # FalkorDB doesn't support parameterized labels, so we whitelist etype
            safe_label = etype if etype in ("Person", "Organization", "Project", "Topic", "Location") else "Entity"
            r.execute_command('GRAPH.QUERY', GRAPH_NAME,
                f"MERGE (n:{safe_label} {{name: $name}}) "
                f"ON CREATE SET n.type = $etype, n.created_at = $ts "
                f"WITH n MATCH (m:Memory {{id: $id}}) MERGE (n)-[:MENTIONED_IN]->(m)",
                '--params', json.dumps({"name": name, "etype": etype, "ts": ts, "id": str(point_id)}))
            connected_entities.append(name)
        except Exception as e:
            print(f"[Graph-Commit] Entity '{name}' error: {e}", flush=True)

    return True, connected_entities

def check_duplicate(vector, text, threshold=0.92):
    """Check if a near-duplicate memory already exists.
    Returns (is_dupe, existing_point_id, similarity) or (False, None, 0)."""
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=3,
            with_payload=True,
        )
        for point in results.points:
            if point.score >= threshold:
                existing_text = point.payload.get("text", "")
                # Also check text overlap to avoid false positives on short generic texts
                words_new = set(text.lower().split())
                words_old = set(existing_text.lower().split())
                overlap = len(words_new & words_old) / max(len(words_new | words_old), 1)
                if overlap > 0.5 or point.score >= 0.95:
                    return True, point.id, round(point.score, 4)
        return False, None, 0
    except Exception as e:
        print(f"[Dedup] Check error: {e}", flush=True)
        return False, None, 0

# ─── A-MAC Admission Control ─────────────────────────────────────────────────

AMAC_THRESHOLD = 4.0
AMAC_OLLAMA_MODEL = os.environ.get("AMAC_MODEL", "qwen3.5:35b")
AMAC_REJECT_LOG = os.environ.get("AMAC_REJECT_LOG", "./logs/amac_rejected.log")
AMAC_TIMEOUT = 30  # seconds — 35B may be busy with swarm agents, give it time

# In-memory metrics (reset on restart)
_amac_metrics = {
    "accepted": 0,
    "rejected": 0,
    "bypassed": 0,
    "score_sum": 0.0,
    "score_count": 0,
    "timeout_accepts": 0,
}

AMAC_PROMPT_TEMPLATE = """You are a memory quality filter for an AI agent. Score the following memory on 3 dimensions.
Return ONLY three integers separated by commas. No text, no explanation, no reasoning. Just three numbers like: 7,4,8

Relevance 0-10: Is this about AI infrastructure, business operations, technology, the user's domain of interest? (0=totally unrelated, 10=highly relevant)
Novelty 0-10: Does this add genuinely NEW, specific information? (0=generic platitude, 10=unique concrete fact)
Specificity 0-10: Is this a concrete verifiable fact with numbers/names/dates? (0=vague filler, 10=specific actionable data)

Examples:
"Things are going well." -> 0,1,0
"BTC went up today." -> 4,2,2
"BrandA DACH revenue hit €580K in Feb 2026, up 23% MoM." -> 10,9,10
"The user's family member had a medical procedure at a major hospital." -> 10,9,10

Memory: "{text}"

Output format: R,N,S (three integers separated by commas, nothing else)
"""

def amac_score(text: str, retry: int = 2):
    """Score text on Relevance, Novelty, Specificity using local llama-swap (OpenAI-compatible).
    Returns (relevance, novelty, specificity, composite) or None on failure/timeout."""
    prompt = AMAC_PROMPT_TEMPLATE.format(text=text[:800])
    
    for attempt in range(retry + 1):
        try:
            resp = requests.post(
                os.environ.get("AMAC_LLM_URL", "http://localhost:11436/v1/chat/completions"),
                json={
                    "model": AMAC_OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.05,  # Lower temperature for more deterministic output
                    "max_tokens": 500,  # Allow full reasoning + score output
                },
                timeout=AMAC_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Robust parsing for any model output style
            # (re imported at module level)
            
            # First, try to find where actual scores are in response
            # The key is finding the FINAL "X,Y,Z" triplet that represents the decision
            
            # Pattern 1: Look for explicit score output after thinking
            # Qwen typically puts results at the very end, often with "Here are" or similar
            lines = raw.split('\n')
            
            # Collect all valid triplets in order
            all_triplets = []
            triplet_positions = []
            
            for idx, line in enumerate(lines):
                line = line.strip()
                scores = re.findall(r'(?<!\d)(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})(?!\d)', line)
                for s in scores:
                    # Validate it looks like actual scores (all 0-10)
                    if all(0 <= int(x) <= 10 for x in s):
                        all_triplets.append(s)
                        triplet_positions.append(idx)
            
            if not all_triplets:
                print(f"[A-MAC] Could not find valid score triplets from: {repr(raw[:200])}", flush=True)
                return None
            
            # Strategy: Use the LAST valid triplet (should be the actual decision, not examples)
            r, n, s = float(all_triplets[-1][0]), float(all_triplets[-1][1]), float(all_triplets[-1][2])
            composite = round((r + n + s) / 3, 2)
            
            # Log what we found for debugging
            print(f"[A-MAC] Parsed scores from triplet #{len(all_triplets)}: r={r}, n={n}, s={s}", flush=True)
            return r, n, s, composite
            
        except requests.exceptions.Timeout:
            print("[A-MAC] Ollama timeout — fail-open, accepting commit", flush=True)
            return None
        except Exception as e:
            if attempt < retry:
                print(f"[A-MAC] Scoring error (attempt {attempt+1}), retrying: {e}", flush=True)
                continue
            print(f"[A-MAC] Scoring error — fail-open: {e}", flush=True)
            return None
    
    return None

def amac_gate(text: str, source: str = "unknown", force: bool = False):
    """A-MAC admission control gate.
    Returns (allowed: bool, reason: str, scores: dict).
    If force=True, bypasses the gate."""
    global _amac_metrics

    if force:
        _amac_metrics["bypassed"] += 1
        return True, "bypassed", {}

    # Skip A-MAC for health check / diagnostic messages - they're system tests, not actual memories
    # (re imported at module level)
    if re.search(r'PIPELINE_TEST_|HEALTH_CHECK_|SYSTEM_DIAGNOSTIC_|memory health check|self-diagnostic', text, re.IGNORECASE):
        return True, "diagnostic_skip", {}

    scores = amac_score(text)

    if scores is None:
        # Fail-open: Ollama unavailable/timeout
        _amac_metrics["accepted"] += 1
        _amac_metrics["timeout_accepts"] += 1
        return True, "timeout_failopen", {}

    relevance, novelty, specificity, composite = scores
    score_dict = {
        "relevance": relevance,
        "novelty": novelty,
        "specificity": specificity,
        "composite": composite,
    }

    # Update metrics
    _amac_metrics["score_sum"] += composite
    _amac_metrics["score_count"] += 1

    if composite >= AMAC_THRESHOLD:
        _amac_metrics["accepted"] += 1
        return True, "accepted", score_dict
    else:
        _amac_metrics["rejected"] += 1
        # Log rejection
        try:
            import datetime
            entry = {
                "ts": datetime.datetime.now().isoformat(),
                "source": source,
                "scores": score_dict,
                "text": text[:200],
            }
            with open(AMAC_REJECT_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as log_err:
            print(f"[A-MAC] Failed to write reject log: {log_err}", flush=True)
        return False, "rejected", score_dict

def commit_memory(text, source="conversation", importance=60, metadata=None, agent_id=None):
    """Commit a memory to Qdrant + FalkorDB graph with inline deduplication.
    If a near-duplicate exists (cosine > 0.92 + text overlap > 0.5), updates instead of creating."""
    try:
        vector = get_embedding(text[:4000], prefix="search_document: ")
    except Exception as e:
        print(f"[commit_memory] Embedding failed, aborting commit: {e}", flush=True)
        return {"ok": False, "error": f"Embedding failed: {e}"}

    # Reject garbage embeddings (e.g. Ollama mid-model-swap returns near-zero vectors)
    import math
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude < 0.1:
        print(f"[commit_memory] REJECTED: embedding magnitude {magnitude:.4f} too low (garbage vector)", flush=True)
        return {"ok": False, "error": f"Embedding magnitude too low: {magnitude:.4f}"}

    import hashlib
    from datetime import datetime
    
    # Inline dedup check
    is_dupe, existing_id, similarity = check_duplicate(vector, text)
    dedup_action = "created"
    
    if is_dupe and existing_id is not None:
        # Update existing point instead of creating new one
        point_id = existing_id
        dedup_action = "updated"
        print(f"[Dedup] Near-duplicate found (sim={similarity}), updating point {point_id}", flush=True)
    else:
        point_id = abs(int(hashlib.md5((text + str(time.time())).encode()).hexdigest()[:15], 16))
    
    timestamp = datetime.now().isoformat()

    payload = {
        "text": text[:4000],
        "source": source,
        "date": timestamp,
        "importance": importance,
        "auto_committed": True,
        "retrieval_count": 0,
    }
    if agent_id:
        payload["agent_id"] = agent_id
    if metadata and isinstance(metadata, dict):
        payload.update(metadata)

    try:
        from qdrant_client.models import PointStruct
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # FalkorDB graph write (non-blocking — don't fail commit if graph is down)
    graph_ok = False
    graph_entities = 0
    connected_to = []
    try:
        entities = extract_entities_fast(text)
        graph_entities = len(entities)
        if entities:
            graph_result = write_to_graph(point_id, text, entities, timestamp)
            if isinstance(graph_result, tuple):
                graph_ok, connected_to = graph_result
            else:
                # Fallback for old return format
                graph_ok = graph_result
        else:
            graph_ok = True  # No entities to write is not a failure
    except Exception as e:
        print(f"[Graph-Commit] Error (non-fatal): {e}", flush=True)

    # Update Qdrant payload with connected_to field if we have connected entities
    if connected_to:
        try:
            qdrant.set_payload(
                collection_name=COLLECTION,
                points=[point_id],
                payload={"connected_to": connected_to}
            )
        except Exception as e:
            print(f"[Graph-Commit] Failed to update connected_to in Qdrant: {e}", flush=True)

    return {
        "ok": True, "id": point_id, "source": source,
        "dedup": {"action": dedup_action, "similarity": similarity if is_dupe else None},
        "graph": {"written": graph_ok, "entities": graph_entities, "connected_to": connected_to}
    }

def _parse_date(date_str):
    """Parse various date formats, return datetime or None."""
    from datetime import datetime as _dt
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.strptime(date_str[:26], fmt)
        except ValueError:
            continue
    return None

def apply_temporal_decay(results, half_life_days=30):
    """Ebbinghaus power-law decay with importance-scaled half-lives.
    
    Changes from linear decay:
    - Power-law: R = e^(-t/S) where S = stability (scales with importance)
    - Important memories (importance >= 80) get 365-day half-life
    - Medium (40-79) get 60-day half-life  
    - Low (<40) get 14-day half-life
    - Retrieval count boosts stability (spaced repetition for AI)
    - Floor at 20% to never fully kill old critical memories
    """
    import math
    from datetime import datetime
    
    now = datetime.now()
    
    for r in results:
        dt = _parse_date(r.get("date", ""))
        if not dt:
            continue
        
        days_old = max((now - dt).total_seconds() / 86400, 0)
        
        # Importance-scaled half-life
        importance = r.get("importance", 50)
        if importance is None:
            importance = 50
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 50
        
        if importance >= 80:
            base_half_life = 365
        elif importance >= 40:
            base_half_life = 60
        else:
            base_half_life = 14
        
        # Retrieval count boost: each retrieval adds 10% to half-life
        retrieval_count = r.get("retrieval_count", 0) or 0
        effective_half_life = base_half_life * (1 + 0.1 * min(retrieval_count, 20))
        
        # Ebbinghaus power-law decay
        stability = effective_half_life / math.log(2)
        decay_factor = math.exp(-days_old / stability)
        
        r["original_score"] = r["score"]
        r["score"] = round(r["score"] * (0.2 + 0.8 * decay_factor), 4)
        r["days_old"] = round(days_old, 1)
        r["effective_half_life"] = round(effective_half_life, 0)
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def apply_multifactor_scoring(results):
    """Multi-factor importance scoring: combines vector similarity, importance,
    recency, source reliability, and retrieval frequency into a composite score.
    
    Formula: score = vector_sim * (0.35 + 0.25*importance_norm + 0.20*recency + 0.10*source_weight + 0.10*retrieval_boost)
    """
    SOURCE_WEIGHTS = {
        "conversation": 0.9, "fact_extractor": 0.85, "chatgpt": 0.8,
        "perplexity": 0.75, "email": 0.6, "telegram": 0.7, "whatsapp": 0.65,
        "social_intel": 0.5, "web_page": 0.4, "benchmark_test": 0.1,
    }
    
    for r in results:
        importance = r.get("importance", 50)
        if importance is None:
            importance = 50
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 50
        importance_norm = min(importance / 100.0, 1.0)
        
        # Source reliability
        source = r.get("source", "")
        source_weight = SOURCE_WEIGHTS.get(source, 0.5)
        # Match partial source names (grok_social_intel_*)
        if source_weight == 0.5 and "social_intel" in source:
            source_weight = 0.5
        
        # Retrieval frequency boost (capped at 1.0)
        retrieval_count = r.get("retrieval_count", 0) or 0
        retrieval_boost = min(retrieval_count / 10.0, 1.0)
        
        # Recency is already embedded in the score via temporal decay,
        # but we give an additional small boost for very recent items
        days_old = r.get("days_old", 30)
        if days_old is not None and days_old < 7:
            recency_bonus = 1.0
        elif days_old is not None and days_old < 30:
            recency_bonus = 0.7
        else:
            recency_bonus = 0.4
        
        # Composite multiplier
        multiplier = 0.35 + 0.25 * importance_norm + 0.20 * recency_bonus + 0.10 * source_weight + 0.10 * retrieval_boost
        
        r["pre_multifactor_score"] = r["score"]
        r["score"] = round(r["score"] * multiplier, 4)
        r["multifactor"] = round(multiplier, 3)
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def qdrant_search(query, limit=10, source_filter=None, agent_id=None):
    """Vector similarity search via Qdrant with temporal decay + multi-factor scoring."""
    try:
        vector = get_embedding(query)
    except Exception as e:
        print(f"[Qdrant] Embedding error: {e}", file=sys.stderr)
        return []

    search_filter = None
    filter_conditions = []
    if source_filter:
        filter_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    if agent_id:
        filter_conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
    if filter_conditions:
        search_filter = Filter(must=filter_conditions)

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=limit,
        query_filter=search_filter,
        with_payload=True,
    )

    out = []
    for point in results.points:
        out.append({
            "score": round(point.score, 4),
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", ""),
            "date": point.payload.get("date", ""),
            "title": point.payload.get("title", ""),
            "url": point.payload.get("url", ""),
            "domain": point.payload.get("domain", ""),
            "importance": point.payload.get("importance", 50),
            "retrieval_count": point.payload.get("retrieval_count", 0),
            "agent_id": point.payload.get("agent_id"),
            "origin": "qdrant",
        })
    
    # Stage 1: Ebbinghaus temporal decay
    out = apply_temporal_decay(out)
    # Stage 2: Multi-factor importance scoring
    out = apply_multifactor_scoring(out)
    return out

# ─── Known Entity Dictionaries (loaded at startup for search-side extraction) ───
# Maps lowercase name/alias → (canonical_name, label_type)
_KNOWN_ENTITY_LOOKUP = {}

def _build_entity_lookup():
    """Build lookup from entity_graph.json + hardcoded known entities."""
    global _KNOWN_ENTITY_LOOKUP
    lookup = {}
    
    # Hardcoded high-value entities with aliases
    _persons = {
        "User": ["user", "user agent"],
        "Partner": ["partner", "spouse"],
        "Family1": ["family_member_1"],
        "Family2": ["family_member_2"],
        "Family3": ["family_member_3"],
        "Contact4": ["contact_4"],
        "Contact5": ["contact_5"],
        "Contact6": ["contact_6"],
        "Contact7": ["contact_7"],
        "Contact8": ["contact_8"],
        "Contact9": ["contact_9"],
        "Contact10": ["contact_10"],
        "Contact11": ["contact_11"],
    }
    _orgs = {
        "BrandA": ["brand_a"],
        "BrandB": ["brand_b"],
        "BrandC": ["brand_c"],
        "BrandD": ["brand_d"],
        "BrandE": ["brand_e"],
        "HoldingCo": ["holding_co"],
        "ParentCo": ["parent_co"],
        "Platform": ["platform", "platform powered"],

        "OpenClaw": ["openclaw"],
        "FalkorDB": ["falkordb"],
        "Qdrant": ["qdrant"],
    }
    _projects = {
        "memory-system": ["memory system"],
        "hybrid_brain": ["hybrid_brain", "hybrid brain", "second brain"],
    }
    _topics = {
        "Medical": ["medical", "health", "treatment"],
        "Business ops": ["revenue", "marketing", "operations"],
        "GPU infrastructure": ["gpu", "inference server", "llama-swap"],
        "Wellness": ["wellness", "supplements", "fitness"],
    }
    
    for name, aliases in _persons.items():
        for a in aliases:
            lookup[a] = (name, "Person")
    for name, aliases in _orgs.items():
        for a in aliases:
            lookup[a] = (name, "Organization")
    for name, aliases in _projects.items():
        for a in aliases:
            lookup[a] = (name, "Project")
    for name, aliases in _topics.items():
        for a in aliases:
            lookup[a] = (name, "Topic")
    
    # Load entity_graph.json if available
    try:
        eg_path = os.path.join(os.path.dirname(__file__), "..", "memory", "entity_graph.json")
        with open(eg_path) as f:
            eg = json.load(f)
        for name in eg.get("people", {}):
            # Handle "Name (alias)" style
            clean = name.split("(")[0].strip()
            for variant in [name.lower(), clean.lower()]:
                if variant not in lookup:
                    lookup[variant] = (clean, "Person")
            # Also add parenthetical aliases
            if "(" in name:
                alias = name.split("(")[1].rstrip(")")
                if alias.lower() not in lookup:
                    lookup[alias.lower()] = (clean, "Person")
        for name in eg.get("companies", {}):
            if name.lower() not in lookup:
                lookup[name.lower()] = (name, "Organization")
        for name in eg.get("topics", {}):
            if name.lower() not in lookup:
                lookup[name.lower()] = (name, "Topic")
        print(f"[HybridBrain] Loaded {len(lookup)} known entity mappings (incl. entity_graph.json)", flush=True)
    except Exception as e:
        print(f"[HybridBrain] entity_graph.json load failed (non-fatal): {e}", flush=True)
    
    _KNOWN_ENTITY_LOOKUP = lookup

_build_entity_lookup()

def extract_entities(query):
    """Smart entity extraction using known entity dictionaries + fallback regex.
    Returns list of (entity_name, label_type) tuples."""
    results = []
    seen = set()
    query_lower = query.lower()
    
    # Phase 1: Match known entities (longest match first to handle multi-word names)
    sorted_keys = sorted(_KNOWN_ENTITY_LOOKUP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in query_lower:
            canonical, label = _KNOWN_ENTITY_LOOKUP[key]
            if canonical not in seen:
                seen.add(canonical)
                results.append((canonical, label))
    
    # Phase 2: Fallback — capitalized words not already matched
    words = query.split()
    i = 0
    while i < len(words):
        word = words[i]
        clean = word.lower().strip(".,!?;:'\"()[]{}").rstrip("'s")
        if clean in STOP_WORDS:
            i += 1
            continue
        if word[0:1].isupper() or word.startswith("@"):
            entity_parts = [word.strip('"@')]
            j = i + 1
            while j < len(words) and words[j][0:1].isupper():
                entity_parts.append(words[j])
                j += 1
            name = " ".join(entity_parts)
            if name not in seen:
                seen.add(name)
                results.append((name, "Unknown"))
            i = j
        else:
            # Non-capitalized content words as keyword fallback (no type)
            if len(clean) > 3 and clean not in STOP_WORDS and clean not in seen:
                seen.add(clean)
                results.append((clean, "Keyword"))
            i += 1
    
    return results

def _decode(val):
    """Decode bytes or return str."""
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return str(val) if val is not None else ""

def graph_search(query, hops=2, limit=10):
    """Graph traversal search via FalkorDB with type-aware entity targeting + 2-hop.
    
    Returns Memory nodes with .text property (first-class reranker candidates)
    plus entity relationship context.
    """
    if FALKORDB_DISABLED:
        return []
    try:
        r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT)
        r.ping()
    except Exception as e:
        print(f"[Graph] Connection error: {e}", file=sys.stderr)
        return []

    typed_entities = extract_entities(query)
    if not typed_entities:
        typed_entities = [(query, "Unknown")]

    results = []
    seen_memory_ids = set()
    seen_context = set()

    for entity_name, entity_type in typed_entities[:4]:
        
        # Determine which labels to search based on entity type
        if entity_type == "Person":
            labels = ["Person"]
        elif entity_type == "Organization":
            labels = ["Organization"]
        elif entity_type == "Project":
            labels = ["Project"]
        elif entity_type == "Topic":
            labels = ["Topic"]
        elif entity_type == "Keyword":
            labels = []  # Keywords skip entity node lookup, go straight to Memory text search
        else:
            labels = ["Person", "Organization", "Project", "Topic", "Location"]

        # --- 1-hop: Entity → Memory nodes (direct mentions) ---
        for label in labels:
            try:
                mem_result = r.execute_command('GRAPH.QUERY', 'brain',
                    f"MATCH (n:{label})-[:MENTIONED_IN]->(m:Memory) "
                    f"WHERE toLower(n.name) CONTAINS toLower($name) "
                    f"RETURN m.id, m.text, m.created_at, n.name LIMIT {limit}",
                    '--params', json.dumps({"name": entity_name}))
                for row in (mem_result[1] or []):
                    mid = _decode(row[0])
                    mtext = _decode(row[1])
                    mdate = _decode(row[2])
                    matched_entity = _decode(row[3])
                    if mid not in seen_memory_ids and mtext and len(mtext) > 10:
                        seen_memory_ids.add(mid)
                        results.append({
                            "text": mtext[:500],
                            "entity": matched_entity,
                            "date": mdate,
                            "origin": "graph",
                            "graph_hop": 1,
                            "source": "graph_memory",
                        })
            except Exception as e:
                print(f"[Graph] 1-hop memory error for {label}/{entity_name}: {e}", file=sys.stderr)

        # --- 2-hop: Entity → co-mentioned entities → their Memory nodes ---
        if hops >= 2 and labels:
            for label in labels:
                try:
                    twohop = r.execute_command('GRAPH.QUERY', 'brain',
                        f"MATCH (n:{label})-[:MENTIONED_IN]->(m1:Memory)<-[:MENTIONED_IN]-(co) "
                        f"WHERE toLower(n.name) CONTAINS toLower($name) AND n <> co "
                        f"WITH DISTINCT co, count(m1) AS shared ORDER BY shared DESC LIMIT 5 "
                        f"MATCH (co)-[:MENTIONED_IN]->(m2:Memory) "
                        f"WHERE NOT m2.id IN $seen "
                        f"RETURN m2.id, m2.text, m2.created_at, co.name, labels(co)[0] LIMIT {limit}",
                        '--params', json.dumps({"name": entity_name, "seen": list(seen_memory_ids)}))
                    for row in (twohop[1] or []):
                        mid = _decode(row[0])
                        mtext = _decode(row[1])
                        mdate = _decode(row[2])
                        co_name = _decode(row[3])
                        co_type = _decode(row[4])
                        if mid not in seen_memory_ids and mtext and len(mtext) > 10:
                            seen_memory_ids.add(mid)
                            results.append({
                                "text": mtext[:500],
                                "entity": entity_name,
                                "connected_to": co_name,
                                "node_type": co_type,
                                "date": mdate,
                                "origin": "graph",
                                "graph_hop": 2,
                                "source": "graph_memory",
                            })
                except Exception as e:
                    print(f"[Graph] 2-hop error for {label}/{entity_name}: {e}", file=sys.stderr)

        # --- Fallback for keywords: search Memory.text directly ---
        if entity_type == "Keyword" and len(entity_name) > 3:
            try:
                keyword_result = r.execute_command('GRAPH.QUERY', 'brain',
                    "MATCH (m:Memory) WHERE toLower(m.text) CONTAINS toLower($name) "
                    "RETURN m.id, m.text, m.created_at LIMIT 5",
                    '--params', json.dumps({"name": entity_name}))
                for row in (keyword_result[1] or []):
                    mid = _decode(row[0])
                    mtext = _decode(row[1])
                    mdate = _decode(row[2])
                    if mid not in seen_memory_ids and mtext and len(mtext) > 10:
                        seen_memory_ids.add(mid)
                        results.append({
                            "text": mtext[:500],
                            "entity": entity_name,
                            "date": mdate,
                            "origin": "graph",
                            "graph_hop": 1,
                            "source": "graph_keyword",
                        })
            except Exception as e:
                print(f"[Graph] Keyword search error: {e}", file=sys.stderr)

        # --- Also collect entity relationship context (non-Memory neighbors) ---
        for label in labels:
            try:
                ctx = r.execute_command('GRAPH.QUERY', 'brain',
                    f"MATCH (n:{label})-[rel]-(connected) "
                    f"WHERE toLower(n.name) CONTAINS toLower($name) "
                    f"AND NOT labels(connected)[0] = 'Memory' "
                    f"RETURN labels(connected)[0], connected.name, type(rel), n.name LIMIT 8",
                    '--params', json.dumps({"name": entity_name}))
                for row in (ctx[1] or []):
                    c_type = _decode(row[0])
                    c_name = _decode(row[1])
                    c_rel = _decode(row[2])
                    n_name = _decode(row[3])
                    key = f"{n_name}:{c_name}:{c_rel}"
                    if key not in seen_context:
                        seen_context.add(key)
                        results.append({
                            "text": f"{c_type}: {c_name}",
                            "entity": n_name,
                            "connected_to": c_name,
                            "relationship": c_rel,
                            "node_type": c_type,
                            "origin": "graph",
                            "graph_hop": 1,
                            "source": "graph_context",
                        })
            except Exception:
                continue

    return results[:limit * 2]  # Return more candidates for reranker

GRAPH_API_URL = "http://127.0.0.1:7778"

def enrich_with_graph(results, limit=5):
    """Call graph_api /expand for entities mentioned in top results.
    Non-fatal: returns empty dict if graph_api is down."""
    try:
        # Extract entity names from top results (connected_to field or entity field)
        entity_names = set()
        for r in results[:limit]:
            if r.get("connected_to"):
                for name in (r["connected_to"] if isinstance(r["connected_to"], list) else [r["connected_to"]]):
                    if name and len(name) > 2:
                        entity_names.add(name)
            if r.get("entity"):
                entity_names.add(r["entity"])
            # Also try extracting from text using fast NER
            text = r.get("text", "")
            for name, _ in extract_entities_fast(text[:500]):
                entity_names.add(name)

        if not entity_names:
            return {}

        enrichment = {}
        for name in list(entity_names)[:8]:  # Cap at 8 entities
            try:
                resp = requests.get(
                    f"{GRAPH_API_URL}/expand",
                    params={"entity": name, "hops": 2},
                    timeout=2,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("connections"):
                        enrichment[name] = data["connections"][:10]
            except Exception:
                continue

        return enrichment
    except Exception as e:
        print(f"[HybridBrain] Graph enrichment error (non-fatal): {e}", flush=True)
        return {}

def hybrid_search(query, limit=10, graph_hops=2, source_filter=None, agent_id=None):
    """Combined Qdrant vector + FalkorDB graph + BM25 + Neural reranking search.
    
    Pipeline: Qdrant vector → BM25 rerank → Neural rerank → merge graph → graph_api enrich → return
    """
    t0 = time.time()

    # Fetch more from Qdrant to give rerankers better candidates
    fetch_limit = limit * 4
    qdrant_results = qdrant_search(query, limit=fetch_limit, source_filter=source_filter, agent_id=agent_id)
    # When filtering by agent_id, skip graph search (graph doesn't store agent_id)
    graph_results = [] if agent_id else graph_search(query, hops=graph_hops, limit=limit)

    # Stage 1: BM25 keyword reranking
    bm25_applied = False
    if BM25_AVAILABLE and qdrant_results:
        try:
            qdrant_results = bm25_rerank(query, qdrant_results)
            bm25_applied = True
        except Exception as e:
            print(f"[HybridBrain] BM25 rerank error: {e}", flush=True)

    # Stage 2: Merge graph Memory results with Qdrant results BEFORE reranking
    # Graph results with actual .text content are first-class reranker candidates
    graph_memory_results = [g for g in graph_results if g.get("source") in ("graph_memory", "graph_keyword")]
    graph_context_results = [g for g in graph_results if g.get("source") == "graph_context"]
    
    # Give graph memory results a base score so they can participate in reranking
    for gr in graph_memory_results:
        gr["score"] = 0.5  # Neutral starting score — reranker will determine real rank
    
    # Combine Qdrant + graph memory candidates for unified reranking
    all_candidates = list(qdrant_results[:limit * 2]) + graph_memory_results
    
    # Stage 3: Neural reranking on the COMBINED pool
    neural_applied = False
    reranker_up = is_reranker_available()
    if reranker_up and all_candidates:
        pre_count = len(all_candidates)
        all_candidates = neural_rerank(query, all_candidates[:limit * 3], top_k=limit)
        neural_applied = len(all_candidates) <= pre_count and any(
            r.get("rerank_score") is not None for r in all_candidates
        )
    else:
        all_candidates = sorted(all_candidates, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    merged = all_candidates[:limit]
    
    # Append graph context results (entity relationships) at the end if there's room
    if graph_context_results and len(merged) < limit:
        for gc in graph_context_results[:limit - len(merged)]:
            gc["score"] = 0.1  # Low score — these are context, not search results
            merged.append(gc)

    # Stage 4: GraphRAG enrichment via graph_api (non-fatal)
    graph_enrichment = enrich_with_graph(merged, limit=5)

    elapsed = time.time() - t0

    # Stage 5: Update access tracking for returned results (fire-and-forget)
    _update_access_tracking([r for r in merged if r.get("origin") != "graph"])

    return {
        "query": query,
        "elapsed_ms": round(elapsed * 1000, 1),
        "results": merged,
        "graph_context": graph_results,  # Keep separate for backward compat
        "graph_enrichment": graph_enrichment,  # Deep multi-hop from graph_api
        "stats": {
            "qdrant_hits": len([r for r in merged if r.get("origin") != "graph"]),
            "graph_hits": len(graph_results),
            "graph_merged": len([r for r in merged if r.get("origin") == "graph"]),
            "graph_enriched_entities": len(graph_enrichment),
            "bm25_reranked": bm25_applied,
            "neural_reranked": neural_applied,
        }
    }

# ─── Access Tracking ─────────────────────────────────────────────────────

def _update_access_tracking(results):
    """Update last_accessed and access_count for returned search results.
    Non-blocking — failures are logged but don't affect search."""
    from datetime import datetime as _dt
    now = _dt.now().isoformat()
    
    for r in results:
        # We need the point ID — reconstruct from text hash if not available
        # Qdrant results from query_points have .id on the point objects,
        # but by the time they're in our dict format, we don't have IDs.
        # We'll use set_payload with a filter approach instead.
        pass  # Access tracking needs point IDs — implemented via scroll below

    # Batch approach: search for the texts and update payloads
    # This is intentionally lightweight — we update in a background thread
    import threading
    
    def _do_update():
        for r in results[:10]:  # Cap at 10 to avoid slow updates
            text = r.get("text", "")
            if not text or len(text) < 10:
                continue
            try:
                # Find the point by searching with its own embedding
                # More efficient: use the text to find the exact point
                search_results = qdrant.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="text", match=MatchValue(value=text[:200]))
                    ]),
                    limit=1,
                    with_payload=False,
                )
                points, _ = search_results
                if points:
                    pid = points[0].id
                    current_count = r.get("retrieval_count", 0) or 0
                    qdrant.set_payload(
                        collection_name=COLLECTION,
                        points=[pid],
                        payload={
                            "last_accessed": now,
                            "access_count": current_count + 1,
                            "retrieval_count": current_count + 1,
                        }
                    )
            except Exception:
                pass  # Non-fatal
    
    try:
        t = threading.Thread(target=_do_update, daemon=True)
        t.start()
    except Exception:
        pass

# ─── Proactive Surfacing ─────────────────────────────────────────────────

def proactive_surface(context_messages, max_results=3):
    """Surface relevant memories the user might want to know about.
    
    Takes recent conversation context, extracts entities/topics,
    searches for related memories NOT already in the conversation,
    and returns high-importance, non-obvious results.
    
    Args:
        context_messages: list of recent message strings (last 2-3)
        max_results: max suggestions to return
    
    Returns:
        list of {"text": ..., "relevance": ..., "reason": ...}
    """
    t0 = time.time()
    
    if not context_messages:
        return []
    
    # Combine context
    full_context = " ".join(context_messages[-3:])
    context_lower = full_context.lower()
    
    # Extract entities and topics from context
    entities = extract_entities(full_context)
    
    # Build diverse search queries from entities
    search_queries = []
    for entity_name, entity_type in entities[:5]:
        if entity_type not in ("Keyword",):
            search_queries.append(entity_name)
    
    # Also add the full context as a query
    if len(full_context) > 20:
        # Use a summary of the context
        words = full_context.split()
        # Pick content words (skip stop words)
        content_words = [w for w in words if w.lower().strip(".,!?") not in STOP_WORDS and len(w) > 3]
        if content_words:
            search_queries.append(" ".join(content_words[:8]))
    
    if not search_queries:
        return []
    
    # Search for each query, collect unique results
    all_results = {}
    
    for query in search_queries[:4]:
        try:
            search_result = hybrid_search(query, limit=5)
            for r in search_result.get("results", []):
                text = r.get("text", "")
                if not text or len(text) < 20:
                    continue
                
                # Novelty check: is this information already obvious from context?
                text_lower = text.lower()
                
                # Skip if >40% of the text words appear in the context
                text_words = set(text_lower.split())
                context_words_set = set(context_lower.split())
                overlap = len(text_words & context_words_set) / max(len(text_words), 1)
                if overlap > 0.4:
                    continue
                
                # Skip very short or generic results
                if len(text) < 30:
                    continue
                
                # Compute a proactive relevance score
                importance = r.get("importance", 50) or 50
                try:
                    importance = int(importance)
                except (ValueError, TypeError):
                    importance = 50
                
                rerank_score = r.get("rerank_score", r.get("score", 0.5))
                
                # Importance-weighted relevance
                proactive_score = rerank_score * 0.5 + (importance / 100) * 0.3 + (1 - overlap) * 0.2
                
                # Only surface high-importance or highly relevant results
                if proactive_score < 0.3:
                    continue
                
                # Deduplicate by text prefix
                key = text[:100]
                if key not in all_results or all_results[key]["proactive_score"] < proactive_score:
                    all_results[key] = {
                        "text": text[:500],
                        "proactive_score": round(proactive_score, 3),
                        "source": r.get("source", ""),
                        "date": r.get("date", ""),
                        "matched_query": query,
                        "overlap": round(overlap, 2),
                        "importance": importance,
                    }
        except Exception as e:
            print(f"[Proactive] Search error for '{query}': {e}", flush=True)
    
    # Sort by proactive score, return top N
    sorted_results = sorted(all_results.values(), key=lambda x: x["proactive_score"], reverse=True)
    
    # Final filtering: ensure diversity (no two results about same entity)
    final = []
    seen_entities = set()
    
    for r in sorted_results:
        r_entities = extract_entities_fast(r["text"])
        r_entity_names = {name.lower() for name, _ in r_entities}
        
        # Skip if all entities already covered
        if r_entity_names and r_entity_names.issubset(seen_entities):
            continue
        
        seen_entities.update(r_entity_names)
        final.append({
            "text": r["text"],
            "relevance": r["proactive_score"],
            "source": r["source"],
            "reason": f"Related to: {r['matched_query']}",
        })
        
        if len(final) >= max_results:
            break
    
    elapsed = time.time() - t0
    print(f"[Proactive] {len(final)} suggestions from {len(search_queries)} queries ({elapsed*1000:.0f}ms)", flush=True)
    
    return final

# ─────────────────────────────────────────────
# HTTP Server
# ─────────────────────────────────────────────
# Optional bearer token auth
MEMORY_API_TOKEN = os.environ.get("MEMORY_API_TOKEN", "")

class HybridHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _check_auth(self):
        """Check bearer token if MEMORY_API_TOKEN is set. Returns True if authorized."""
        if not MEMORY_API_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {MEMORY_API_TOKEN}":
            return True
        self._send_json({"error": "Unauthorized"}, 401)
        return False

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/search":
            query = params.get("q", [""])[0]
            limit = int(params.get("limit", ["10"])[0])
            source = params.get("source", [None])[0]
            agent_id = params.get("agent_id", [None])[0] if TENANT_MODE else None

            if not query:
                self._send_json({"error": "Missing q parameter"}, 400)
                return

            result = hybrid_search(query, limit=limit, source_filter=source, agent_id=agent_id)
            self._send_json(result)

        elif parsed.path == "/graph":
            query = params.get("q", [""])[0]
            limit = int(params.get("limit", ["10"])[0])
            hops = int(params.get("hops", ["2"])[0])

            if not query:
                self._send_json({"error": "Missing q parameter"}, 400)
                return

            results = graph_search(query, hops=hops, limit=limit)
            self._send_json({"query": query, "results": results})

        elif parsed.path == "/stats":
            try:
                info = qdrant.get_collection(COLLECTION)
                qdrant_count = info.points_count
            except Exception:
                qdrant_count = -1

            try:
                r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT)
                node_count = r.execute_command('GRAPH.QUERY', 'brain', 'MATCH (n) RETURN count(n)')[1][0][0]
                edge_count = r.execute_command('GRAPH.QUERY', 'brain', 'MATCH ()-[e]->() RETURN count(e)')[1][0][0]
            except Exception:
                node_count = -1
                edge_count = -1

            self._send_json({
                "qdrant": {"collection": COLLECTION, "points": qdrant_count},
                "graph": {"nodes": node_count, "edges": edge_count},
                "status": "ok"
            })

        elif parsed.path == "/health":
            health = {
                "status": "ok",
                "engine": "hybrid-brain",
                "version": "5.0",
                "components": {
                    "qdrant": "unknown",
                    "falkordb": "unknown",
                    "ollama_embed": "unknown",
                    "reranker": "up" if is_reranker_available() else "down",
                    "bm25": "up" if BM25_AVAILABLE else "down",
                }
            }
            # Check Qdrant
            try:
                info = qdrant.get_collection(COLLECTION)
                health["components"]["qdrant"] = f"up ({info.points_count} pts)"
            except Exception:
                health["components"]["qdrant"] = "down"
                health["status"] = "degraded"
            # Check FalkorDB
            try:
                r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT)
                r.ping()
                health["components"]["falkordb"] = "up"
            except Exception:
                health["components"]["falkordb"] = "down"
                health["status"] = "degraded"
            # Check Ollama embeddings
            try:
                test_resp = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": "health check"}, timeout=5)
                if test_resp.status_code == 200 and "embeddings" in test_resp.json():
                    health["components"]["ollama_embed"] = "up"
                else:
                    health["components"]["ollama_embed"] = "error"
                    health["status"] = "degraded"
            except Exception:
                health["components"]["ollama_embed"] = "down"
                health["status"] = "degraded"
            self._send_json(health)

        elif parsed.path == "/amac/metrics":
            total = _amac_metrics["accepted"] + _amac_metrics["rejected"]
            avg_score = (
                round(_amac_metrics["score_sum"] / _amac_metrics["score_count"], 2)
                if _amac_metrics["score_count"] > 0 else None
            )
            rejection_rate = round(_amac_metrics["rejected"] / total * 100, 1) if total > 0 else 0
            self._send_json({
                "accepted": _amac_metrics["accepted"],
                "rejected": _amac_metrics["rejected"],
                "bypassed": _amac_metrics["bypassed"],
                "timeout_accepts": _amac_metrics["timeout_accepts"],
                "total": total,
                "avg_composite_score": avg_score,
                "rejection_rate_pct": rejection_rate,
                "threshold": AMAC_THRESHOLD,
            })

        else:
            self._send_json({"error": f"Unknown path: {parsed.path}"}, 404)

    def do_POST(self):
        if not self._check_auth():
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        parsed = urlparse(self.path)

        if parsed.path == "/search":
            query = data.get("q", data.get("query", ""))
            limit = data.get("limit", 10)
            source = data.get("source", None)
            agent_id = data.get("agent_id") if TENANT_MODE else None

            if not query:
                self._send_json({"error": "Missing query"}, 400)
                return

            result = hybrid_search(query, limit=limit, source_filter=source, agent_id=agent_id)
            self._send_json(result)

        elif parsed.path == "/proactive":
            messages = data.get("messages", [])
            context = data.get("context", "")
            max_results = data.get("max_results", 3)

            # Accept either a list of messages or a single context string
            if context and not messages:
                messages = [context]
            
            if not messages:
                self._send_json({"error": "Missing messages or context"}, 400)
                return

            suggestions = proactive_surface(messages, max_results=max_results)
            self._send_json({
                "suggestions": suggestions,
                "count": len(suggestions),
            })

        elif parsed.path == "/commit":
            text = data.get("text", "")
            source = data.get("source", "conversation")
            importance = data.get("importance", 60)
            metadata = data.get("metadata", None)
            force = bool(data.get("force", False))
            agent_id = data.get("agent_id") if TENANT_MODE else None

            if not text:
                self._send_json({"error": "Missing text"}, 400)
                return

            # ── A-MAC admission gate ──
            allowed, reason, scores = amac_gate(text, source=source, force=force)
            if not allowed:
                self._send_json({
                    "ok": False,
                    "rejected": True,
                    "reason": "amac_below_threshold",
                    "scores": scores,
                    "threshold": AMAC_THRESHOLD,
                }, 200)
                return
            # ─────────────────────────

            result = commit_memory(text, source=source, importance=importance, metadata=metadata, agent_id=agent_id)
            if scores:
                result["amac"] = {"reason": reason, "scores": scores}
            status = 200 if result.get("ok") else 500
            self._send_json(result, status)

        else:
            self._send_json({"error": f"Unknown path: {parsed.path}"}, 404)

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True

def serve(port=7777):
    server = ReusableHTTPServer(("127.0.0.1", port), HybridHandler)
    print(f"[HybridBrain] Serving on http://127.0.0.1:{port}", flush=True)
    print(f"[HybridBrain] Qdrant: {COLLECTION} ({qdrant.get_collection(COLLECTION).points_count} pts)", flush=True)
    try:
        r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT)
        nc = r.execute_command('GRAPH.QUERY', 'brain', 'MATCH (n) RETURN count(n)')[1][0][0]
        ec = r.execute_command('GRAPH.QUERY', 'brain', 'MATCH ()-[e]->() RETURN count(e)')[1][0][0]
        print(f"[HybridBrain] FalkorDB: {nc} nodes, {ec} edges", flush=True)
    except Exception:
        print("[HybridBrain] FalkorDB: unavailable (graph search disabled)", flush=True)
    server.serve_forever()

def run_tests():
    print("🧪 Hybrid Brain Test Suite")
    print("=" * 50)

    test_queries = [
        "Brand withdrawal complaint",
        "user business operations",
        "inference server GPU",
        "partner wedding",
        "proxy model routing",
        "follistatin gene therapy",
        "Brazil gambling regulation",
    ]

    for q in test_queries:
        result = hybrid_search(q, limit=3)
        print(f'\nQuery: "{q}"')
        print(f'  Qdrant: {result["stats"]["qdrant_hits"]} hits | Graph: {result["stats"]["graph_hits"]} hits | {result["elapsed_ms"]}ms')
        if result['results']:
            top = result['results'][0]
            txt = top['text'][:100].replace('\n', ' ')
            print(f'  Top Qdrant: [{top["source"]}] score={top["score"]} | {txt}...')
        if result['graph_context']:
            top_g = result['graph_context'][0]
            print(f'  Top Graph: {top_g.get("entity","")} --{top_g.get("relationship","")}-- {top_g.get("connected_to","")}')

    print("\n" + "=" * 50)
    print("✅ Tests complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Brain Search Server")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    global TENANT_MODE
    if args.tenant_mode:
        TENANT_MODE = True

    if args.test:
        run_tests()
    else:
        serve(args.port)
