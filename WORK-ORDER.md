# RASPUTIN Memory v3.0 — Complete Work Order

**Generated:** 2026-03-30
**Target Executor:** OpenCode Zen (Claude Opus)
**Source:** 13 audit reports (6 × 122B domain audits + 6 × Opus cross-exams + 1 master architecture review)
**Repository:** https://github.com/jcartu/rasputin-memory

---

## Preamble

### What Is rasputin-memory?

A hybrid AI memory system ("second brain") combining:
- **Qdrant** (port 6333) — 127K+ vectors, 768d, nomic-embed-text
- **FalkorDB** (port 6380) — Knowledge graph, 240K nodes, 535K edges
- **BM25** — Keyword scoring layer
- **Neural Reranker** — BGE-reranker-v2-m3 (port 8006)
- **A-MAC** — LLM-powered admission quality gate
- **Ebbinghaus temporal decay** — Memory forgetting curve
- **OpenClaw hook** (hooks/openclaw-mem/) — Auto-recall, session management

### File Tree (Key Files)

```
rasputin-memory/
├── tools/
│   ├── hybrid_brain.py          # Core API server (1573 lines) — port 7777
│   ├── memory_engine.py         # CLI search pipeline (867 lines) — DUPLICATE, to be merged
│   ├── bm25_search.py           # BM25 keyword scoring (143 lines)
│   ├── reranker_server.py       # Neural reranker wrapper (123 lines)
│   ├── fact_extractor.py        # Cron: mines sessions for facts (602 lines)
│   ├── memory_consolidator_v4.py # Parallel consolidation (479 lines)
│   ├── memory_decay.py          # Decay/archive engine (452 lines)
│   ├── memory_dedup.py          # Dedup tool (354 lines)
│   ├── embed_server_gpu1.py     # Embedding server wrapper
│   ├── enrich_second_brain.py   # Overnight enrichment cron
│   ├── memory_health_check.py   # Health check script
│   ├── memory_mcp_server.py     # MCP protocol wrapper (stub)
│   ├── hybrid_brain_v2_tenant.py  # DEAD — delete
│   ├── memory_consolidate.py      # DEAD — superseded by v4
│   ├── smart_memory_query.py      # DEAD — delete
│   ├── consolidate_second_brain.py # One-time migration
│   └── memory_autogen.py         # Dev utility
├── graph-brain/
│   ├── graph_api.py             # FalkorDB REST API (268 lines) — port 7778
│   ├── schema.py                # Graph schema setup
│   ├── graph_query.py           # DEAD — delete
│   └── migrate_to_graph.py      # DEAD — already run
├── hooks/openclaw-mem/
│   ├── handler.js               # OpenClaw hook (1521 lines)
│   ├── database.js              # SQLite observations
│   ├── context-builder.js       # Session context injection
│   └── gateway-llm.js           # LLM calls for summarization
├── brainbox/                    # DEAD — Hebbian memory, never integrated
├── predictive-memory/           # DEAD — never integrated
├── storm-wiki/                  # DEAD — research toy
├── honcho/                      # DEAD — broken integration
├── config/
│   └── known_entities.json
├── tests/
│   ├── test_smoke.py            # 15 parse-only tests
│   └── conftest.py              # Minimal
├── benchmarks/
│   └── README.md                # Placeholder only
├── docker-compose.yml
├── Dockerfile
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
├── quickstart.sh
└── Makefile
```

### Dependencies & Services

| Service | Port | How to Start |
|---------|------|-------------|
| Qdrant | 6333 | `docker compose up -d qdrant` |
| FalkorDB | 6380 | `docker compose up -d falkordb` |
| Ollama (embeddings) | 11434 | `systemctl start ollama` |
| Memory API (hybrid_brain) | 7777 | `python3 tools/hybrid_brain.py` |
| Graph API | 7778 | `python3 graph-brain/graph_api.py` |
| Neural Reranker | 8006 | `python3 tools/reranker_server.py` |

### How to Run Tests

```bash
pytest tests/ -v                    # Current smoke tests
pytest tests/ -v --cov=tools        # With coverage (after adding real tests)
```

### How to Start Dev Server

```bash
docker compose up -d                # Start Qdrant, FalkorDB, Redis
python3 tools/hybrid_brain.py       # Start memory API on :7777
```

### Git Workflow

- **Commit after each phase** with conventional commits
- Format: `feat(phase-N): description` or `fix(phase-N): description`
- Push after each phase
- Branch: work directly on `main` (single developer)

---

## Phase 1: Critical Bug Fixes

**Goal:** Fix every confirmed bug that causes data loss, silent failures, or dead features. These are all independent fixes — no architecture changes needed.

**Expected outcome:** All 7 layers of the hybrid search pipeline actually work. Decay system operates on the correct collection. Fact extractor stops double-committing. Concurrency is safe.

---

### Task 1: Fix Decay Engine Collection Name
**Priority:** P0-CRITICAL
**Source:** 03-maintenance-lifecycle.md, opus-03-maintenance-crossexam.md
**Validated by:** Both agree (122B flagged HIGH, Opus escalated to CRITICAL)
**File(s):** `tools/memory_decay.py`
**Line(s):** 32
**Current behavior:** `COLLECTION = "memories_v2"` — decay engine scans/archives from a collection that doesn't exist in production. The entire decay system is a no-op. Brain grows unbounded.
**Expected behavior:** Decay engine targets `"second_brain"` (the live production collection).
**Fix description:**
Change line 32:
```python
# OLD
COLLECTION = "memories_v2"
# NEW
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
```
Also add `import os` at the top if not already present.
**Tests to add:** `test_decay_collection_matches_production()` — verify COLLECTION == "second_brain"
**Dependencies:** None

---

### Task 2: Define BM25_AVAILABLE Variable
**Priority:** P0-CRITICAL
**Source:** opus-01-retrieval-crossexam.md, MEGA-REPORT.md
**Validated by:** Opus only (122B analyzed BM25 without noticing it was dead)
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~30 (after the bm25 import), referenced at ~1046, ~1388
**Current behavior:** `BM25_AVAILABLE` is referenced but never defined. Every search that reaches the BM25 codepath raises `NameError`. BM25+RRF fusion has NEVER worked in the HTTP API.
**Expected behavior:** BM25 is enabled when the import succeeds.
**Fix description:**
After the existing import block (around line 30):
```python
from bm25_search import hybrid_rerank as bm25_rerank
print("[HybridBrain] BM25 reranking: enabled", flush=True)
```
Add:
```python
BM25_AVAILABLE = True
```
If the import is in a try/except, set `BM25_AVAILABLE = False` in the except block.
**Tests to add:** `test_bm25_available_defined()` — verify the variable exists after import
**Dependencies:** None

---

### Task 3: Fix ASCII-Only Regex (Russian Text Invisible)
**Priority:** P0-CRITICAL
**Source:** 01-retrieval-pipeline.md, opus-05-tests-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/bm25_search.py`, `tools/memory_engine.py`
**Line(s):** bm25_search.py:~17, memory_engine.py:~91, memory_engine.py:~186
**Current behavior:** Tokenizer uses `re.findall(r'[a-zA-Z0-9]+', text.lower())` which strips ALL Cyrillic/non-ASCII characters. Russian-language memories are invisible to BM25 search, OM lookup, and graph traversal.
**Expected behavior:** Unicode word characters are preserved.
**Fix description:**
In all three locations, change:
```python
# OLD
re.findall(r'[a-zA-Z0-9]+', text.lower())
# NEW
re.findall(r'[\w]+', text.lower(), re.UNICODE)
```
Or equivalently `r'\w+'` since `re.UNICODE` is default in Python 3.
**Tests to add:** `test_bm25_tokenizer_cyrillic()` — verify "Москва" tokenizes correctly
**Dependencies:** None

---

### Task 4: Remove Fact Extractor Double-Commit
**Priority:** P0-CRITICAL
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/fact_extractor.py`
**Line(s):** 392-428
**Current behavior:** Every extracted fact is written TWICE: (1) direct Qdrant REST API with UUID5 string ID and no embed prefix, (2) POST to `/commit` endpoint with MD5 int ID and proper prefix. Two copies per fact with different embeddings.
**Expected behavior:** Facts are committed only once via the `/commit` endpoint (which handles dedup, A-MAC, graph).
**Fix description:**
Delete the direct Qdrant write block (lines ~392-413). Keep only the `/commit` POST (lines ~420-428). The `/commit` endpoint already handles embedding, dedup, A-MAC gating, and graph writes.
**Tests to add:** `test_fact_extractor_single_commit()` — verify only one write path exists
**Dependencies:** None

---

### Task 5: Fix Fact Extractor Embed Endpoint and Prefix
**Priority:** P1-HIGH
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/fact_extractor.py`
**Line(s):** 28, 393-395
**Current behavior:** Defaults to `/api/embeddings` (Ollama's older endpoint) with `"prompt"` key and NO `"search_document: "` prefix. All other files use `/api/embed` with `"input"` key and the prefix.
**Expected behavior:** All embedding calls use the same endpoint and prefix convention.
**Fix description:**
Line 28: Change default to `/api/embed`:
```python
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
```
If there's still a direct embedding call (after Task 4 removes the direct Qdrant write, this may be moot), ensure it uses `"input"` key with `"search_document: "` prefix.
**Tests to add:** `test_embed_url_consistency()` — verify all files use same endpoint
**Dependencies:** Task 4 (may make this partially moot)

---

### Task 6: Add Threading Lock to commit_memory
**Priority:** P1-HIGH
**Source:** 02-ingestion-pipeline.md, opus-02-ingestion-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** Around the `commit_memory()` function definition
**Current behavior:** `ThreadingHTTPServer` processes concurrent POST requests. `commit_memory()` has no locking. Two simultaneous commits of similar text can both pass the dedup check and create duplicates.
**Expected behavior:** Commit is serialized to prevent race conditions.
**Fix description:**
Add at module level:
```python
import threading
_commit_lock = threading.Lock()
```
Wrap the body of `commit_memory()` (or at least the dedup-check + upsert section):
```python
def commit_memory(text, source="conversation", importance=60, metadata=None, force=False):
    with _commit_lock:
        # existing body
```
**Tests to add:** `test_concurrent_commits_no_dupes()` — submit 10 identical commits in parallel, verify only 1 stored
**Dependencies:** None

---

### Task 7: Fix Thread-Unsafe _amac_metrics
**Priority:** P1-HIGH
**Source:** opus-06-infrastructure-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** Around line ~270 (where `_amac_metrics` dict is defined)
**Current behavior:** `_amac_metrics["accepted"] += 1` from multiple threads without locking. `+=` on dict values is not atomic — read-modify-write race condition.
**Expected behavior:** Metrics are thread-safe.
**Fix description:**
Replace the plain dict with thread-safe counters:
```python
import threading
_amac_metrics_lock = threading.Lock()
_amac_metrics = {"accepted": 0, "rejected": 0, "timeout": 0, "error": 0, "forced": 0}

def _inc_metric(key):
    with _amac_metrics_lock:
        _amac_metrics[key] += 1
```
Replace all `_amac_metrics["key"] += 1` with `_inc_metric("key")`.
**Tests to add:** `test_amac_metrics_thread_safe()` — concurrent metric increments produce correct totals
**Dependencies:** None

---

### Task 8: Fix Access Tracking (Carry Point IDs Through Pipeline)
**Priority:** P1-HIGH
**Source:** 03-maintenance-lifecycle.md, opus-01-retrieval-crossexam.md, opus-06-infrastructure-crossexam.md
**Validated by:** Both agree (multiple reports)
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~1111-1187 (`_update_access_tracking`), and wherever search results are constructed
**Current behavior:** Three compounding failures: (1) `text[:200]` exact match via `MatchValue` never matches stored `text[:4000]`, (2) reads stale `retrieval_count` from search dict (lost-update race), (3) caps at 10 updates. Access tracking is silently a no-op.
**Expected behavior:** After search, correctly increment `retrieval_count` and `last_accessed` for returned results.
**Fix description:**
1. When constructing search result dicts from Qdrant responses, include `"point_id": point.id` in each result dict
2. Rewrite `_update_access_tracking` to use point IDs directly:
```python
def _update_access_tracking(results):
    now = datetime.now().isoformat()
    def _do_update():
        for r in results:
            pid = r.get("point_id")
            if not pid:
                continue
            try:
                # Read current count from Qdrant (not from stale search result)
                points = qdrant.retrieve(collection_name=COLLECTION, ids=[pid], with_payload=True)
                if points:
                    current_count = (points[0].payload or {}).get("retrieval_count", 0) or 0
                    qdrant.set_payload(
                        collection_name=COLLECTION,
                        points=[pid],
                        payload={
                            "last_accessed": now,
                            "retrieval_count": current_count + 1,
                        }
                    )
            except Exception:
                pass
    try:
        t = threading.Thread(target=_do_update, daemon=True)
        t.start()
    except Exception:
        pass
```
3. Remove the cap at 10 — update all returned results
**Tests to add:** `test_access_tracking_increments()` — commit a memory, search for it, verify retrieval_count increases
**Dependencies:** None

---

### Task 9: Add Importance Floor to Soft-Delete
**Priority:** P1-HIGH
**Source:** 03-maintenance-lifecycle.md, opus-03-maintenance-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/memory_decay.py`
**Line(s):** ~206-220 (where soft-delete candidates are selected)
**Current behavior:** Memories >180 days since last access go to soft-delete regardless of importance. Critical memories (importance ≥ 80) can be deleted.
**Expected behavior:** Memories with high importance are protected from soft-delete.
**Fix description:**
In the soft-delete candidate selection, add importance check:
```python
if days_since_access >= SOFT_DELETE_DAYS:
    importance = payload.get("importance", 50)
    if isinstance(importance, (int, float)) and importance >= 80:
        stats["protected_high_importance"] += 1
        continue  # Skip soft-delete for critical memories
    stats["soft_delete_candidates"] += 1
    soft_delete_candidates.append(...)
```
**Tests to add:** `test_high_importance_not_soft_deleted()` — memory with importance=95 and 200 days age is not a soft-delete candidate
**Dependencies:** Task 1 (decay must target correct collection first)

---

### Task 10: Fix Non-Atomic Archive-Then-Delete
**Priority:** P1-HIGH
**Source:** 03-maintenance-lifecycle.md, opus-03-maintenance-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/memory_decay.py`
**Line(s):** ~267-298
**Current behavior:** Upsert to archive collection, then delete from main collection. Crash between = data loss.
**Expected behavior:** Two-phase approach prevents data loss.
**Fix description:**
Implement a "mark-then-sweep" approach:
1. Set a `"pending_archive": true` flag on the point in the main collection
2. Upsert to archive collection
3. Delete from main collection
4. On startup, check for any `pending_archive` points and complete the operation
```python
# Phase 1: Mark
qdrant.set_payload(collection_name=COLLECTION, points=batch_ids,
    payload={"pending_archive": True, "archive_started": datetime.now().isoformat()})

# Phase 2: Archive
qdrant.upsert(collection_name=ARCHIVE_COLLECTION, points=archive_points)

# Phase 3: Delete
qdrant.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=batch_ids))
```
**Tests to add:** `test_archive_atomicity()` — verify no data loss on simulated crash
**Dependencies:** Task 1

---

### Task 11: Route memory_engine.py commit() Through /commit API
**Priority:** P1-HIGH
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/memory_engine.py`
**Line(s):** ~656-681 (the `commit()` function)
**Current behavior:** `memory_engine.py commit` writes directly to Qdrant REST API bypassing A-MAC, dedup, graph writes. Uses Python's randomized `hash()` for point IDs.
**Expected behavior:** All commits go through the unified `/commit` endpoint.
**Fix description:**
Replace the direct Qdrant write with an HTTP POST:
```python
def commit(text, source="conversation", importance=60, metadata=None):
    resp = requests.post("http://localhost:7777/commit",
        json={"text": text, "source": source, "importance": importance, "metadata": metadata},
        timeout=30)
    return resp.json()
```
**Tests to add:** `test_memory_engine_commit_uses_api()` — verify commit goes through HTTP
**Dependencies:** None

---

### Task 12: Route consolidator_v4 Through /commit API
**Priority:** P1-HIGH
**Source:** opus-03-maintenance-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/memory_consolidator_v4.py`
**Line(s):** `commit_to_brain()` function (~line 180)
**Current behavior:** Bypasses A-MAC quality gate, dedup, graph writes. No `importance` field, uses `timestamp` instead of `date`. Magnitude threshold too low (0.1 vs needed 0.8).
**Expected behavior:** Consolidated facts go through the standard commit pipeline.
**Fix description:**
Replace the direct Qdrant write with:
```python
def commit_to_brain(text, source="consolidator_v4", metadata=None):
    try:
        resp = requests.post("http://localhost:7777/commit",
            json={"text": text, "source": source, "importance": 60, "metadata": metadata},
            timeout=30)
        return resp.json()
    except Exception as e:
        print(f"[Consolidator-v4] Commit error: {e}", flush=True)
        return {"ok": False, "error": str(e)}
```
Remove the direct Qdrant REST API write and embedding code from this file.
**Tests to add:** `test_consolidator_uses_commit_api()`
**Dependencies:** None

---

### Task 13: Fix handler.js URL Templates
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Opus only (master review)
**File(s):** `hooks/openclaw-mem/handler.js`
**Line(s):** All lines with `${MEMORY_API_URL:-...}` style strings
**Current behavior:** JavaScript template literals use bash-style `${VAR:-default}` syntax which doesn't work in JS. These resolve to literal strings, meaning auto-recall and Honcho fetch calls go to invalid URLs. Auto-recall is silently broken.
**Expected behavior:** Proper `process.env` reads with defaults.
**Fix description:**
Find all instances like:
```javascript
// OLD
const url = `${MEMORY_API_URL:-http://localhost:7777}/search`;
// NEW
const MEMORY_API_URL = process.env.MEMORY_API_URL || 'http://localhost:7777';
const url = `${MEMORY_API_URL}/search`;
```
Define the env var reads once at the top of the file and use them throughout. Also fix any Honcho URL templates the same way.
**Tests to add:** Manual verification that auto-recall actually calls localhost:7777
**Dependencies:** None

---

### Task 14: Fix Word-Boundary Matching for Known Entities
**Priority:** P2-MEDIUM
**Source:** opus-04-graph-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~193-207 (`extract_entities_fast`)
**Current behavior:** Uses `if name.lower() in text_lower` which is substring containment. Entity "Al" matches "also", "algorithm". Entity "PM" matches "PM2", "3PM".
**Expected behavior:** Word-boundary matching only.
**Fix description:**
```python
# OLD
if name.lower() in text_lower:
# NEW
if re.search(r'\b' + re.escape(name.lower()) + r'\b', text_lower):
```
**Tests to add:** `test_entity_extraction_no_substring_match()` — "Al" should not match "algorithm"
**Dependencies:** None

---

### Task 15: Fix Dedup Tokenizer (Strip Punctuation)
**Priority:** P2-MEDIUM
**Source:** opus-05-tests-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~274-277 (inside `check_duplicate` text overlap calculation)
**Current behavior:** `text.lower().split()` doesn't strip punctuation. "hello," and "hello" are different tokens. Jaccard coefficient is artificially low, near-dupes slip through.
**Expected behavior:** Consistent tokenizer that strips punctuation.
**Fix description:**
```python
# OLD
words1 = set(text1.lower().split())
words2 = set(text2.lower().split())
# NEW
words1 = set(re.findall(r'\w+', text1.lower()))
words2 = set(re.findall(r'\w+', text2.lower()))
```
**Tests to add:** `test_dedup_ignores_punctuation()` — "hello, world!" and "hello world" should have high overlap
**Dependencies:** None

---

### Task 16: Fix Temporal Decay Timezone Handling
**Priority:** P2-MEDIUM
**Source:** opus-05-tests-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~562-563 (`apply_temporal_decay`) and ~536 (`_parse_date`)
**Current behavior:** `datetime.now()` returns naive local time. `_parse_date` truncates timezone info with `date_str[:26]`. If server timezone changes, all decay shifts.
**Expected behavior:** UTC-aware timestamps throughout.
**Fix description:**
```python
# In apply_temporal_decay:
from datetime import timezone
now = datetime.now(timezone.utc)

# In _parse_date — don't truncate timezone:
# Handle both naive and aware datetimes, normalize to UTC
```
**Tests to add:** `test_temporal_decay_timezone_consistent()`
**Dependencies:** None

---

### Task 17: Add Text Length Validation on /commit
**Priority:** P2-MEDIUM
**Source:** opus-02-ingestion-crossexam.md, 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** `/commit` handler (around line ~870-920)
**Current behavior:** `if not text` is the only check. Empty string rejected, but `"ok"` (2 chars) passes. No max length check — 10MB text accepted.
**Expected behavior:** Text must be 20-8000 characters.
**Fix description:**
```python
if not text or len(text.strip()) < 20:
    self._send_json({"error": "Text too short (min 20 chars)"}, 400)
    return
if len(text) > 8000:
    self._send_json({"error": "Text too long (max 8000 chars)"}, 400)
    return
```
**Tests to add:** `test_commit_rejects_short_text()`, `test_commit_rejects_long_text()`
**Dependencies:** None

---

### Task 18: Add Importance Range Validation on /commit
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md, opus-06-infrastructure-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** `/commit` handler
**Current behavior:** `importance = data.get("importance", 60)` — any value accepted, including strings and negatives.
**Expected behavior:** Integer 0-100.
**Fix description:**
```python
importance = data.get("importance", 60)
try:
    importance = int(importance)
    importance = max(0, min(100, importance))
except (ValueError, TypeError):
    importance = 60
```
**Tests to add:** `test_commit_clamps_importance()`
**Dependencies:** None

---

### Task 19: Add Request Size Limit
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** `do_POST` method, where Content-Length is read
**Current behavior:** No cap on request body size. Client can send 100MB.
**Expected behavior:** Max 1MB request body.
**Fix description:**
```python
MAX_BODY_SIZE = 1 * 1024 * 1024  # 1MB

def do_POST(self):
    content_length = int(self.headers.get("Content-Length", 0))
    if content_length > MAX_BODY_SIZE:
        self._send_json({"error": "Request too large"}, 413)
        return
    body = self.rfile.read(content_length)
    ...
```
**Tests to add:** `test_rejects_oversized_request()`
**Dependencies:** None

---

### Task 20: Protect Payload Fields from Metadata Overwrite
**Priority:** P2-MEDIUM
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~484-486 (where `payload.update(metadata)` is called)
**Current behavior:** Caller can pass `metadata={"text": "evil", "source": "admin"}` and overwrite core payload fields.
**Expected behavior:** Protected fields cannot be overwritten.
**Fix description:**
```python
PROTECTED_FIELDS = {"text", "source", "date", "importance", "embedding_model", "schema_version",
                    "retrieval_count", "last_accessed", "point_id"}

if metadata and isinstance(metadata, dict):
    safe_metadata = {k: v for k, v in metadata.items() if k not in PROTECTED_FIELDS}
    payload.update(safe_metadata)
```
**Tests to add:** `test_metadata_cannot_overwrite_text()`
**Dependencies:** None

---

### Task 21: Fix A-MAC Prompt Example Pollution
**Priority:** P2-MEDIUM
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only (122B noted fragility but not mechanism)
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~309-313 (AMAC_PROMPT_TEMPLATE) and ~354 (regex parser)
**Current behavior:** The prompt includes numeric examples (`0,1,0` / `4,2,2` / `10,9,10`) that the regex parser can match. With "last triplet" strategy and `max_tokens: 500`, verbose models may echo examples after the real answer.
**Expected behavior:** Parser reliably extracts the actual score.
**Fix description:**
Add a sentinel marker to the prompt:
```python
# In AMAC_PROMPT_TEMPLATE, after examples, add:
"Reply with ONLY the three numbers separated by commas. SCORES:"
```
Then modify the parser to look for text after "SCORES:" first:
```python
# Check for sentinel
if "SCORES:" in response_text:
    response_text = response_text.split("SCORES:")[-1]
# Then apply existing regex
```
**Tests to add:** `test_amac_parser_ignores_examples()`
**Dependencies:** None

---

### Task 22: Cache _load_known_entities with TTL
**Priority:** P2-MEDIUM
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~191 (`extract_entities_fast` calls `_load_known_entities()`)
**Current behavior:** Reads and parses JSON file from disk on every commit. 10 commits/sec = 10 file reads/sec.
**Expected behavior:** Cache with 5-minute TTL.
**Fix description:**
```python
_entities_cache = {"data": None, "ts": 0}

def _load_known_entities():
    if time.time() - _entities_cache["ts"] < 300:
        return _entities_cache["data"]
    # ... existing file load logic ...
    _entities_cache["data"] = result
    _entities_cache["ts"] = time.time()
    return result
```
**Tests to add:** `test_entity_cache_ttl()`
**Dependencies:** None

---

### Task 23: Fix Docker Port Exposure
**Priority:** P2-MEDIUM
**Source:** opus-06-infrastructure-crossexam.md
**Validated by:** Opus only
**File(s):** `docker-compose.yml`
**Line(s):** brain service ports section
**Current behavior:** `"7777:7777"` exposes to all interfaces, bypassing the Python server's localhost binding.
**Expected behavior:** Bind to localhost only.
**Fix description:**
```yaml
# OLD
ports:
  - "7777:7777"
# NEW
ports:
  - "127.0.0.1:7777:7777"
```
**Tests to add:** N/A (config change)
**Dependencies:** None

---

### Task 24: Fix Reranker max_length
**Priority:** P3-LOW
**Source:** 01-retrieval-pipeline.md
**Validated by:** Both agree
**File(s):** `tools/reranker_server.py`
**Line(s):** ~84
**Current behavior:** `max_length=512` truncates long memories before reranking.
**Expected behavior:** `max_length=1024` for better reranking of longer texts.
**Fix description:**
```python
# OLD
max_length=512
# NEW
max_length=1024
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 25: Wire A-MAC Scores to Importance Field
**Priority:** P2-MEDIUM
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** Where A-MAC scores are computed but discarded
**Current behavior:** A-MAC computes Relevance, Novelty, Specificity scores but they're not used to set the `importance` field. Importance stays at the caller-provided value (default 60).
**Expected behavior:** A-MAC composite score influences the stored importance.
**Fix description:**
After A-MAC scoring, blend with caller-provided importance:
```python
if amac_scores:
    amac_composite = amac_scores["composite"]
    # Scale 0-10 composite to 0-100, blend with provided importance
    amac_importance = amac_composite * 10
    importance = int(0.4 * importance + 0.6 * amac_importance)
```
**Tests to add:** `test_amac_influences_importance()`
**Dependencies:** None

---

---

## Phase 2: Pipeline Unification

**Goal:** Merge `hybrid_brain.py` and `memory_engine.py` into a single search pipeline. Extract `memory_engine.py`'s superior query expansion and source tiering into the unified system. Delete the duplicate.

**Expected outcome:** One search path, not two. Every search goes through the same pipeline with all features (query expansion + temporal decay + BM25 + graph + reranking).

---

### Task 26: Extract Query Expansion from memory_engine.py
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/memory_engine.py` (source), create new `tools/pipeline/query_expansion.py`
**Current behavior:** `memory_engine.py` has sophisticated 12-angle query expansion that `hybrid_brain.py` lacks.
**Expected behavior:** Query expansion is a standalone module importable by hybrid_brain.
**Fix description:**
1. Create `tools/pipeline/` directory with `__init__.py`
2. Extract `expand_queries()` and related helper functions from `memory_engine.py` into `tools/pipeline/query_expansion.py`
3. Clean up the interface: `def expand_queries(query: str, max_expansions: int = 5) -> list[str]`
4. Import and use in `hybrid_brain.py`'s search handler
**Tests to add:** `test_query_expansion_basic()`, `test_query_expansion_entity_aware()`
**Dependencies:** None

---

### Task 27: Extract Source Tiering from memory_engine.py
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/memory_engine.py` (source), create `tools/pipeline/source_tiering.py`
**Current behavior:** `memory_engine.py` has Gold/Silver/Bronze source tiers. `hybrid_brain.py` has flat source weights.
**Expected behavior:** Source tiering is a standalone module used in the unified pipeline.
**Fix description:**
1. Extract source tier definitions and scoring logic into `tools/pipeline/source_tiering.py`
2. Define tiers: Gold (email, perplexity, chatgpt), Silver (conversation, telegram), Bronze (auto-extract, consolidator)
3. Expose: `def get_source_weight(source: str) -> float`
4. Integrate into hybrid_brain.py's multi-factor scoring
**Tests to add:** `test_source_tiering_gold()`, `test_source_tiering_unknown_source()`
**Dependencies:** None

---

### Task 28: Integrate Query Expansion into hybrid_brain.py Search
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/hybrid_brain.py`
**Line(s):** Search handler
**Current behavior:** hybrid_brain.py uses the raw query for vector search. No expansion.
**Expected behavior:** Queries are expanded before embedding, results from all angles are merged.
**Fix description:**
In `hybrid_search()`:
```python
from pipeline.query_expansion import expand_queries

def hybrid_search(query, limit=10, ...):
    # Expand query into multiple search angles
    queries = expand_queries(query, max_expansions=5)
    
    # Search with each expanded query
    all_results = []
    for q in queries:
        embedding = get_embedding(q, prefix=EMBED_QUERY_PREFIX)
        results = qdrant.query_points(...)
        all_results.extend(results)
    
    # Deduplicate by point_id, keep highest score
    # ... then proceed with existing BM25, reranking, etc.
```
**Tests to add:** `test_search_uses_query_expansion()`
**Dependencies:** Tasks 26, 27

---

### Task 29: Convert memory_engine.py to Thin CLI Client
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/memory_engine.py`
**Current behavior:** 867-line standalone search implementation with its own embedding, dedup, ranking.
**Expected behavior:** Thin CLI that calls `http://localhost:7777/search` and `http://localhost:7777/commit`.
**Fix description:**
Rewrite `memory_engine.py` to:
```python
#!/usr/bin/env python3
"""Memory Engine CLI — thin client for the Memory API."""
import requests
import sys
import json

API_URL = os.environ.get("MEMORY_API_URL", "http://localhost:7777")

def recall(query, limit=10):
    resp = requests.get(f"{API_URL}/search", params={"q": query, "limit": limit}, timeout=30)
    return resp.json()

def commit(text, source="conversation", importance=60):
    resp = requests.post(f"{API_URL}/commit",
        json={"text": text, "source": source, "importance": importance}, timeout=30)
    return resp.json()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 memory_engine.py recall|commit <text>")
        sys.exit(1)
    cmd, text = sys.argv[1], " ".join(sys.argv[2:])
    if cmd == "recall":
        result = recall(text)
        for r in result.get("results", []):
            print(f"[{r.get('score', 0):.3f}] {r.get('text', '')[:200]}")
    elif cmd == "commit":
        result = commit(text)
        print(json.dumps(result, indent=2))
```
This removes ~800 lines of duplicate code.
**Tests to add:** `test_memory_engine_cli_search()`, `test_memory_engine_cli_commit()`
**Dependencies:** Tasks 26, 27, 28

---

---

## Phase 3: Architecture Cleanup

**Goal:** Consolidate configuration, add type hints, improve error handling, delete dead code, add schema versioning.

**Expected outcome:** Single config source. All dead code removed. ~3000 lines of code deleted.

---

### Task 30: Create config/rasputin.toml
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** Create `config/rasputin.toml`, modify `tools/hybrid_brain.py`
**Current behavior:** 9+ config sources: env vars, hardcoded constants, JSON files, broken JS templates.
**Expected behavior:** Single TOML config file with env var overrides.
**Fix description:**
Create `config/rasputin.toml`:
```toml
[server]
port = 7777
host = "127.0.0.1"

[qdrant]
url = "http://localhost:6333"
collection = "second_brain"

[graph]
host = "localhost"
port = 6380
graph_name = "brain"
disabled = false

[embeddings]
url = "http://localhost:11434/api/embed"
model = "nomic-embed-text"
prefix_query = "search_query: "
prefix_doc = "search_document: "

[reranker]
url = "http://localhost:8006/rerank"
timeout = 15
enabled = true

[amac]
threshold = 4.0
timeout = 30
model = "qwen2.5:14b"

[scoring]
decay_half_life_low = 14
decay_half_life_medium = 60
decay_half_life_high = 365

[entities]
known_entities_path = "config/known_entities.json"
```

Create `tools/config.py`:
```python
import os
import tomllib

def load_config(path="config/rasputin.toml"):
    with open(path, "rb") as f:
        config = tomllib.load(f)
    # Env var overrides
    config["qdrant"]["url"] = os.environ.get("QDRANT_URL", config["qdrant"]["url"])
    config["qdrant"]["collection"] = os.environ.get("QDRANT_COLLECTION", config["qdrant"]["collection"])
    # ... etc for all overridable values
    return config
```
Then update `hybrid_brain.py` to import and use `load_config()` instead of scattered env var reads.
**Tests to add:** `test_config_loads()`, `test_config_env_override()`
**Dependencies:** None

---

### Task 31: Add Schema Versioning to All New Commits
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/hybrid_brain.py`
**Line(s):** `commit_memory()` function, where payload is constructed
**Current behavior:** No `embedding_model` or `schema_version` in payload. 127K vectors with no way to know which model created them.
**Expected behavior:** Every new commit tagged with embedding model and schema version.
**Fix description:**
In `commit_memory()`, add to payload:
```python
payload = {
    "text": text,
    "source": source,
    "date": timestamp,
    "importance": importance,
    "retrieval_count": 0,
    "embedding_model": EMBED_MODEL,       # NEW
    "schema_version": "3.0",              # NEW
    # ... existing fields
}
```
**Tests to add:** `test_commit_includes_schema_version()`
**Dependencies:** None

---

### Task 32: Backfill Existing Vectors with Schema Metadata
**Priority:** P2-MEDIUM
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** Create `tools/backfill_schema.py`
**Current behavior:** 127K existing vectors have no `embedding_model` or `schema_version`.
**Expected behavior:** All vectors tagged with `{"embedding_model": "nomic-embed-text", "schema_version": "2.0"}`.
**Fix description:**
Create a script that scrolls through all points and sets the payload:
```python
from qdrant_client import QdrantClient
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "second_brain"

offset = None
batch_size = 100
total = 0

while True:
    results, offset = qdrant.scroll(
        collection_name=COLLECTION, offset=offset, limit=batch_size,
        with_payload=False, with_vectors=False)
    if not results:
        break
    ids = [p.id for p in results]
    qdrant.set_payload(collection_name=COLLECTION, points=ids,
        payload={"embedding_model": "nomic-embed-text", "schema_version": "2.0"})
    total += len(ids)
    print(f"Backfilled {total} points")
```
**Tests to add:** Run the script and verify with a sample check
**Dependencies:** Task 31

---

### Task 33: Delete Dead Code Files
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** Multiple
**Current behavior:** ~3000 lines of dead/duplicate code cluttering the repo.
**Expected behavior:** Only production code remains.
**Fix description:**
Delete these files:
1. `tools/hybrid_brain_v2_tenant.py` — dead fork (identical minus tenant filter)
2. `tools/memory_consolidate.py` — superseded by v4
3. `tools/smart_memory_query.py` — duplicates memory_engine
4. `graph-brain/graph_query.py` — duplicates graph_api
5. `graph-brain/migrate_to_graph.py` — one-time script, already run
6. `brainbox/` — entire directory (never integrated)
7. `predictive-memory/` — entire directory (never integrated)
8. `storm-wiki/` — entire directory (research toy)
9. `honcho/` — entire directory (broken integration, dead code)

Also remove the `run_tests()` function from hybrid_brain.py (lines ~1535-1570) — it's not a real test.
**Tests to add:** `test_no_dead_imports()` — verify nothing imports deleted files
**Dependencies:** Task 29 (memory_engine.py must be converted to thin client first)

---

### Task 34: Add Type Hints to hybrid_brain.py Core Functions
**Priority:** P2-MEDIUM
**Source:** 05-tests-benchmarks.md, opus-05-tests-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Current behavior:** ~80% of functions lack type hints.
**Expected behavior:** All public functions have type annotations.
**Fix description:**
Add type hints to at minimum these functions:
```python
def hybrid_search(query: str, limit: int = 10, graph_hops: int = 2,
                  source_filter: Optional[str] = None) -> dict: ...
def commit_memory(text: str, source: str = "conversation",
                  importance: int = 60, metadata: Optional[dict] = None,
                  force: bool = False) -> dict: ...
def get_embedding(text: str, prefix: str = "") -> list[float]: ...
def batch_embed(texts: list[str], prefix: str = "") -> list[list[float]]: ...
def neural_rerank(query: str, results: list[dict], top_k: int = 10) -> list[dict]: ...
def check_duplicate(text: str, embedding: list[float]) -> Optional[dict]: ...
def amac_score(text: str) -> Optional[dict]: ...
def graph_search(query: str, limit: int = 10) -> list[dict]: ...
def write_to_graph(point_id: int, text: str, entities: list[tuple], timestamp: str) -> bool: ...
def apply_temporal_decay(results: list[dict]) -> list[dict]: ...
```
**Tests to add:** Run `mypy tools/hybrid_brain.py --ignore-missing-imports` in CI
**Dependencies:** None

---

### Task 35: Add Type Hints to bm25_search.py
**Priority:** P2-MEDIUM
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** `tools/bm25_search.py`
**Current behavior:** No type hints on public functions.
**Expected behavior:** Full type annotations.
**Fix description:**
```python
def hybrid_rerank(query: str, results: list[dict], bm25_weight: float = 0.3) -> list[dict]: ...
def reciprocal_rank_fusion(dense_results: list[dict], bm25_scores: list[float],
                            k: int = 60) -> list[dict]: ...
```
**Tests to add:** mypy check
**Dependencies:** None

---

### Task 36: Add Graceful Shutdown Signal Handlers
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** `serve()` function
**Current behavior:** No signal handling. SIGTERM kills active requests mid-flight.
**Expected behavior:** Graceful shutdown on SIGTERM/SIGINT.
**Fix description:**
```python
import signal

def serve(port=7777):
    server = ReusableHTTPServer(("127.0.0.1", port), HybridHandler)
    
    def shutdown_handler(signum, frame):
        print(f"[HybridBrain] Shutting down gracefully...", flush=True)
        server.shutdown()
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    print(f"[HybridBrain] Serving on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
```
**Tests to add:** N/A (signal handling)
**Dependencies:** None

---

### Task 37: Add Global Exception Handler to HTTP Server
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Current behavior:** Unhandled exceptions crash threads, may expose stack traces.
**Expected behavior:** All exceptions caught and returned as 500 with safe error message.
**Fix description:**
Wrap `do_GET` and `do_POST` methods:
```python
def do_POST(self):
    try:
        self._handle_post()
    except Exception as e:
        print(f"[HybridBrain] Unhandled error: {e}", flush=True)
        try:
            self._send_json({"error": "Internal server error"}, 500)
        except:
            pass

def _handle_post(self):
    # ... existing do_POST logic moved here
```
**Tests to add:** `test_server_handles_unexpected_error()`
**Dependencies:** None

---

### Task 38: Rename Temporal Decay Comment (Exponential, Not Power-Law)
**Priority:** P3-LOW
**Source:** 03-maintenance-lifecycle.md, opus-03-maintenance-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~741 (comment above decay formula)
**Current behavior:** Comment says "Ebbinghaus power-law decay" but formula is `e^(-t/S)` which is exponential.
**Expected behavior:** Comment accurately describes the formula.
**Fix description:**
```python
# OLD comment
# Ebbinghaus power-law decay
# NEW comment
# Exponential decay (inspired by Ebbinghaus forgetting curve)
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 39: Remove Dead half_life_days Parameter
**Priority:** P3-LOW
**Source:** opus-05-tests-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** Function signature of `apply_temporal_decay`
**Current behavior:** `half_life_days=30` parameter in signature is never used — function always computes half-life from importance tiers.
**Expected behavior:** Remove misleading parameter.
**Fix description:**
Remove `half_life_days` from the function signature. Update any callers if they pass it.
**Tests to add:** N/A
**Dependencies:** None

---

### Task 40: Remove retrieval_count Dead Field Initialization
**Priority:** P3-LOW
**Source:** opus-02-ingestion-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~483
**Current behavior:** `"retrieval_count": 0` initialized at commit time. Before Task 8, this field was never incremented. After Task 8, it will be.
**Expected behavior:** Keep initialization (now it will actually be used after Task 8).
**Fix description:** No change needed — Task 8 fixes the increment. This task is just a note: verify after Task 8 that the field works end-to-end.
**Tests to add:** Covered by Task 8 tests
**Dependencies:** Task 8

---

### Task 41: Consolidate Collection Name References
**Priority:** P2-MEDIUM
**Source:** 03-maintenance-lifecycle.md, opus-03-maintenance-crossexam.md
**Validated by:** Both agree
**File(s):** All files that reference collection names
**Current behavior:** Different collection names hardcoded in different files: `"second_brain"`, `"memories_v2"`, `"second_brain_v2"`.
**Expected behavior:** All files read collection name from config.
**Fix description:**
After Task 30 (config consolidation), ensure ALL files read collection name from config or env var:
```python
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
```
Grep all `.py` files for hardcoded collection names and replace.
**Tests to add:** `test_all_files_use_config_collection()` — grep for hardcoded strings
**Dependencies:** Task 30

---

### Task 42: Add Structured Logging (Replace print statements)
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md, opus-06-infrastructure-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py` and all tools
**Current behavior:** All logging is unstructured `print()` statements. `log_message` in HTTP handler is suppressed entirely.
**Expected behavior:** Python `logging` module with structured JSON output.
**Fix description:**
1. Add at top of hybrid_brain.py:
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("hybrid_brain")
```
2. Replace all `print(f"[HybridBrain] ...")` with `logger.info(...)`, `logger.error(...)`, etc.
3. Re-enable `log_message` in HTTP handler with proper logging:
```python
def log_message(self, fmt, *args):
    logger.debug(f"{self.client_address[0]} {fmt % args}")
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 43: Add Dockerfile Non-Root User
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `Dockerfile`
**Current behavior:** Container runs as root.
**Expected behavior:** Runs as non-root user.
**Fix description:**
Add to Dockerfile:
```dockerfile
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
# ... (after COPY and pip install)
RUN chown -R appuser:appgroup /app
USER appuser
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 44: Add FalkorDB Health Check to docker-compose
**Priority:** P3-LOW
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `docker-compose.yml`
**Current behavior:** FalkorDB has no health check.
**Expected behavior:** Health check enables proper depends_on with condition.
**Fix description:**
```yaml
falkordb:
  image: falkordb/falkordb:latest
  ports:
    - "127.0.0.1:6380:6379"
  volumes:
    - falkordb_data:/data
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "redis-cli", "-p", "6379", "ping"]
    interval: 10s
    timeout: 5s
    retries: 3
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 45: Add BM25 RRF Length Mismatch Guard
**Priority:** P2-MEDIUM
**Source:** opus-05-tests-crossexam.md
**Validated by:** Opus only
**File(s):** `tools/bm25_search.py`
**Line(s):** ~72-103 (`reciprocal_rank_fusion`)
**Current behavior:** Assumes `len(dense_results) == len(bm25_scores)`. Length mismatch causes silent quality degradation or IndexError.
**Expected behavior:** Explicit validation.
**Fix description:**
```python
def reciprocal_rank_fusion(dense_results, bm25_scores, k=60):
    if len(dense_results) != len(bm25_scores):
        # Pad shorter list with zeros
        max_len = max(len(dense_results), len(bm25_scores))
        while len(bm25_scores) < max_len:
            bm25_scores.append(0.0)
        # Truncate if bm25 is longer
        bm25_scores = bm25_scores[:len(dense_results)]
    # ... existing logic
```
**Tests to add:** `test_rrf_handles_length_mismatch()`
**Dependencies:** None

---

---

## Phase 4: Graph Layer Overhaul

**Goal:** Fix the graph layer to be functional and consistent. Unify NER, fix relationship directions, parameterize queries, remove dead graph code.

**Expected outcome:** Graph search actually contributes meaningful results. Entities extracted on write are findable on read.

---

### Task 46: Unify Entity Extraction (Single NER Path)
**Priority:** P1-HIGH
**Source:** opus-04-graph-crossexam.md
**Validated by:** Opus only (found 3 incompatible NER implementations)
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~185 (`extract_entities_fast`), ~785 (`extract_entities`)
**Current behavior:** THREE different NER implementations: (1) `extract_entities_fast()` for write (known entities + capitalized phrases), (2) `extract_entities()` for read (different lookup dict), (3) LLM-based in migrate_to_graph.py (broken). Entities found at commit may not be found at search.
**Expected behavior:** One `extract_entities()` function used for both read and write.
**Fix description:**
1. Keep `extract_entities_fast()` as the unified function (it's the most complete)
2. Apply word-boundary matching (Task 14)
3. Delete `extract_entities()` (the read-side version)
4. Update `graph_search()` to use `extract_entities_fast()` for query entity extraction
5. Ensure both read and write paths produce identical entity sets from identical text
**Tests to add:** `test_ner_consistency()` — same text produces same entities on read and write paths
**Dependencies:** Task 14

---

### Task 47: Fix Relationship Direction Consistency
**Priority:** P1-HIGH
**Source:** 04-graph-knowledge.md, opus-04-graph-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`, `graph-brain/graph_api.py`
**Current behavior:** `hybrid_brain.py` writes `Entity-[:MENTIONED_IN]->Memory`. `migrate_to_graph.py` wrote `Memory-[:MENTIONS]->Entity`. `graph_search()` queries `Entity-[:MENTIONED_IN]->Memory` — so migrated data is invisible.
**Expected behavior:** One consistent direction: `(Memory)-[:MENTIONS]->(Entity)` everywhere.
**Fix description:**
1. In `write_to_graph()` in hybrid_brain.py, change:
```python
# OLD
f"MERGE (n)-[:MENTIONED_IN]->(m)"
# NEW
f"WITH n MATCH (m:Memory {{id: $id}}) MERGE (m)-[:MENTIONS]->(n)"
```
2. In `graph_search()`, change query direction:
```python
# OLD
f"MATCH (n:{label})-[:MENTIONED_IN]->(m:Memory)"
# NEW
f"MATCH (m:Memory)-[:MENTIONS]->(n:{label})"
```
3. In `graph_api.py` `expand_by_node_id`, use `[:MENTIONS|MENTIONED_IN]` for backwards compatibility during transition
4. Create a one-time migration script to reverse existing `MENTIONED_IN` edges to `MENTIONS`:
```python
# Pseudocode for migration
gq("MATCH (n)-[r:MENTIONED_IN]->(m:Memory) "
   "CREATE (m)-[:MENTIONS]->(n) DELETE r")
```
**Tests to add:** `test_graph_write_and_read_consistent()`
**Dependencies:** None

---

### Task 48: Fix Graph Result Scoring
**Priority:** P2-MEDIUM
**Source:** 04-graph-knowledge.md, opus-04-graph-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~597 (where graph results get `score = 0.5`)
**Current behavior:** All graph results get arbitrary `score = 0.5` regardless of relevance.
**Expected behavior:** Graph results have meaningful scores based on hop distance and relationship strength.
**Fix description:**
```python
for gr in graph_memory_results:
    hop_count = gr.get("graph_hop", 1)
    # 1-hop entities score higher than 2-hop
    gr["score"] = 0.8 if hop_count == 1 else 0.5
```
This gives 1-hop results a fighting chance against vector results during neural reranking. The reranker will still sort by semantic relevance from the text.
**Tests to add:** `test_graph_results_scored_by_hop()`
**Dependencies:** None

---

### Task 49: Fix Graph Text Truncation Disadvantage
**Priority:** P2-MEDIUM
**Source:** opus-04-graph-crossexam.md (deeper analysis)
**Validated by:** Opus only
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~892 (where graph results truncate text)
**Current behavior:** Graph results use `mtext[:500]` while Qdrant results have full text. Neural reranker sees less content for graph results, disadvantaging them.
**Expected behavior:** Graph results include full text.
**Fix description:**
Change `mtext[:500]` to use the full text from the Memory node, or fetch the full text from Qdrant using the Memory node's ID.
**Tests to add:** N/A
**Dependencies:** None

---

### Task 50: Add Input Validation to expand_by_node_id
**Priority:** P2-MEDIUM
**Source:** opus-04-graph-crossexam.md
**Validated by:** Opus only
**File(s):** `graph-brain/graph_api.py`
**Line(s):** ~121-135
**Current behavior:** `hops` and `limit` are interpolated into Cypher without internal validation. External callers could pass `hops=100` causing combinatorial explosion.
**Expected behavior:** Validated internally.
**Fix description:**
```python
def expand_by_node_id(node_id: int, hops: int = 2, limit: int = 30):
    hops = min(max(int(hops), 1), 4)
    limit = min(max(int(limit), 1), 100)
    # ... existing query
```
**Tests to add:** `test_expand_clamps_hops()`
**Dependencies:** None

---

### Task 51: Add Redis Reconnection Logic
**Priority:** P2-MEDIUM
**Source:** opus-04-graph-crossexam.md
**Validated by:** Opus only
**File(s):** `graph-brain/graph_api.py`, `tools/hybrid_brain.py`
**Current behavior:** graph_api.py uses a cached singleton Redis connection with no reconnect. If FalkorDB restarts, all queries fail until process restart. hybrid_brain.py creates new connections per request (inefficient but self-healing).
**Expected behavior:** Connection pool with reconnection.
**Fix description:**
In hybrid_brain.py, create a module-level Redis connection pool:
```python
import redis
_redis_pool = redis.ConnectionPool(host=FALKOR_HOST, port=FALKOR_PORT, max_connections=10)

def get_redis():
    return redis.Redis(connection_pool=_redis_pool)
```
In graph_api.py, add a health check before returning cached connection:
```python
def get_redis():
    global _redis
    if _redis is not None:
        try:
            _redis.ping()
            return _redis
        except redis.ConnectionError:
            _redis = None
    _redis = redis.Redis(host=HOST, port=PORT)
    return _redis
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 52: Fix gq() Silent Error Handling
**Priority:** P3-LOW
**Source:** opus-04-graph-crossexam.md
**Validated by:** Opus only
**File(s):** `graph-brain/graph_api.py`
**Line(s):** ~40-49
**Current behavior:** `gq()` returns `[]` for both empty results and errors. No way to distinguish.
**Expected behavior:** Errors are logged and propagated.
**Fix description:**
```python
def gq(cypher: str):
    try:
        raw = r.execute_command("GRAPH.QUERY", GRAPH_NAME, cypher)
        rows = raw[1] if len(raw) > 1 else []
        return rows
    except Exception as e:
        logger.error(f"Graph query error: {e}, query: {cypher[:200]}")
        return []
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 53: Fix escape_cypher in migrate_to_graph.py
**Priority:** P3-LOW
**Source:** opus-04-graph-crossexam.md
**Validated by:** Opus only
**File(s):** `graph-brain/migrate_to_graph.py`
**Line(s):** ~140
**Current behavior:** `escape_cypher()` doesn't handle newlines, carriage returns, or null bytes.
**Expected behavior:** Complete escaping.
**Fix description:**
Note: This file is marked for deletion in Task 33. If it's kept for reference, fix:
```python
def escape_cypher(s):
    return str(s).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\0", "")
```
**Tests to add:** N/A
**Dependencies:** Task 33 may make this moot

---

---

## Phase 5: New Features

**Goal:** Add contradiction detection, importance recalculation, relevance feedback, and proactive context surfacing.

**Expected outcome:** The memory system actively detects conflicts, adapts importance over time, and learns from usage.

---

### Task 54: Implement Contradiction Detection on Commit
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** Create `tools/pipeline/contradiction.py`, modify `tools/hybrid_brain.py`
**Current behavior:** "User moved to Moscow" and "User moved to St. Petersburg" coexist forever with no flag.
**Expected behavior:** On commit, check for semantically similar memories that may contradict. Flag or auto-resolve.
**Fix description:**
1. Create `tools/pipeline/contradiction.py`:
```python
def check_contradictions(text: str, embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search for existing memories that might contradict the new one."""
    similar = qdrant.query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=top_k,
        score_threshold=0.85  # High similarity = potential contradiction
    )
    contradictions = []
    for r in similar:
        # If cosine > 0.85 but texts assert different facts about same entity
        # Use simple heuristic: contains negation or different values for same subject
        if looks_contradictory(text, r.payload.get("text", "")):
            contradictions.append({
                "existing_id": r.id,
                "existing_text": r.payload.get("text", "")[:200],
                "similarity": r.score,
            })
    return contradictions
```
2. In `commit_memory()`, after dedup check, call `check_contradictions()`. If found:
   - Add `"contradicts": [existing_id]` to payload
   - Add `"supersedes": [existing_id]` if this is clearly newer info
   - Log the contradiction for review
3. Add `/contradictions` GET endpoint that lists flagged contradictions
**Tests to add:** `test_contradiction_detected()`, `test_non_contradictory_passes()`
**Dependencies:** Phases 1-2

---

### Task 55: Implement Importance Recalculation Cron
**Priority:** P2-MEDIUM
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** Create `tools/importance_recalculator.py`
**Current behavior:** Importance is static, set once at commit time. A memory about "looking for a car" stays at importance=60 forever.
**Expected behavior:** Daily cron re-evaluates importance based on recency, retrieval frequency, and topic currency.
**Fix description:**
Create `tools/importance_recalculator.py`:
```python
def recalculate_importance():
    """Scan all memories and update importance based on current context."""
    # Scroll through all points
    # For each, compute new importance:
    #   - Boost if retrieval_count > 5 (actively used)
    #   - Decay if no access in 90+ days AND no recent related commits
    #   - Boost if related to recent commits (topic is "hot")
    # Update payload with new importance
```
Add to crontab: daily at 03:00.
**Tests to add:** `test_importance_recalc_boosts_active()`, `test_importance_recalc_decays_stale()`
**Dependencies:** Task 8 (access tracking must work first)

---

### Task 56: Implement Relevance Feedback Endpoint
**Priority:** P2-MEDIUM
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/hybrid_brain.py`
**Current behavior:** No way for the agent to signal "this memory was helpful" or "irrelevant."
**Expected behavior:** `/feedback` endpoint adjusts memory importance and retrieval weight.
**Fix description:**
Add POST `/feedback` endpoint:
```python
elif parsed.path == "/feedback":
    point_id = data.get("point_id")
    helpful = data.get("helpful", True)  # True = helpful, False = irrelevant
    
    points = qdrant.retrieve(collection_name=COLLECTION, ids=[point_id], with_payload=True)
    if points:
        current = points[0].payload
        importance = current.get("importance", 50)
        if helpful:
            importance = min(100, importance + 5)
        else:
            importance = max(0, importance - 10)
        qdrant.set_payload(collection_name=COLLECTION, points=[point_id],
            payload={"importance": importance, "last_feedback": datetime.now().isoformat()})
```
**Tests to add:** `test_feedback_positive_boosts()`, `test_feedback_negative_decays()`
**Dependencies:** Task 8 (point IDs must be in search results)

---

### Task 57: Fix enrich_with_graph (Currently Dead Code)
**Priority:** P2-MEDIUM
**Source:** 04-graph-knowledge.md, opus-04-graph-crossexam.md
**Validated by:** Both agree
**File(s):** `tools/hybrid_brain.py`
**Line(s):** ~623-647
**Current behavior:** 2-second timeout (too short), calls graph_api which isn't running (port 7778), returns `{}` on any error. Enrichment is never actually used in results.
**Expected behavior:** Graph enrichment works and adds context to search results.
**Fix description:**
1. Increase timeout to 10 seconds
2. Instead of calling graph_api (separate server), call graph functions directly (FalkorDB is already connected)
3. Actually use the enrichment in the response:
```python
def enrich_with_graph(results, limit=5):
    enrichment = {}
    for r in results[:limit]:
        entities = extract_entities_fast(r.get("text", ""))
        for name, etype in entities:
            # Get related entities from graph
            related = graph_search_direct(name, limit=3)
            if related:
                enrichment[name] = related
    return enrichment

# In hybrid_search(), add to response:
result["graph_context"] = enrich_with_graph(final_results)
```
**Tests to add:** `test_graph_enrichment_returns_data()`
**Dependencies:** Tasks 46, 47

---

### Task 58: Implement Proactive Context Surfacing
**Priority:** P2-MEDIUM
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `tools/hybrid_brain.py`
**Current behavior:** `/proactive` endpoint exists but nobody calls it. The current implementation is basic.
**Expected behavior:** Proactive surfacing triggered by the OpenClaw hook, providing session-aware context.
**Fix description:**
1. Enhance the `/proactive` endpoint to accept current conversation context:
```python
elif parsed.path == "/proactive":
    # Input: recent message text, time of day, active entities
    recent_text = data.get("context", "")
    entities = extract_entities_fast(recent_text)
    
    # Search for related memories the user hasn't seen recently
    suggestions = []
    for name, etype in entities:
        graph_results = graph_search(name, limit=3)
        for gr in graph_results:
            if gr.get("last_accessed") and (now - parse_date(gr["last_accessed"])).days > 7:
                suggestions.append(gr)
    
    return suggestions[:5]
```
2. In handler.js (after fixing URL templates in Task 13), call `/proactive` at session bootstrap
**Tests to add:** `test_proactive_surfaces_related_memories()`
**Dependencies:** Tasks 13, 46

---

### Task 59: Add Embedding Version Tracking for Future Migration
**Priority:** P2-MEDIUM
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** Create `tools/embedding_health.py`
**Current behavior:** No way to detect if embedding model was changed. Mixed-model vectors produce meaningless cosine distances.
**Expected behavior:** Weekly check validates embedding consistency.
**Fix description:**
Create `tools/embedding_health.py`:
```python
def check_embedding_consistency():
    """Sample 100 random points, re-embed their text, compare to stored vector."""
    sample = qdrant.scroll(collection_name=COLLECTION, limit=100, with_vectors=True, with_payload=True)
    drifted = 0
    for point in sample[0]:
        text = point.payload.get("text", "")
        current_embedding = get_embedding(text, prefix="search_document: ")
        stored = point.vector
        cosine = dot(current_embedding, stored) / (norm(current_embedding) * norm(stored))
        if cosine < 0.95:
            drifted += 1
    return {"total": len(sample[0]), "drifted": drifted, "drift_rate": drifted / len(sample[0])}
```
Add to crontab: weekly.
**Tests to add:** `test_embedding_health_no_drift()`
**Dependencies:** Task 31

---

---

## Phase 6: Test Suite

**Goal:** Build comprehensive tests for all pipeline stages and the overall system.

**Expected outcome:** 80%+ coverage on pipeline modules. Regression prevention. CI runs tests on every push.

---

### Task 60: Create Test Infrastructure (conftest.py)
**Priority:** P0-CRITICAL
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** `tests/conftest.py`
**Current behavior:** conftest.py has only a sys.path manipulation. No fixtures.
**Expected behavior:** Shared fixtures for mocking Qdrant, FalkorDB, Ollama.
**Fix description:**
```python
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

@pytest.fixture
def mock_qdrant(monkeypatch):
    """Mock Qdrant client for unit tests."""
    class MockQdrant:
        def __init__(self):
            self.points = {}
        def query_points(self, **kwargs):
            return []
        def upsert(self, **kwargs):
            pass
        def scroll(self, **kwargs):
            return ([], None)
        def retrieve(self, **kwargs):
            return []
        def set_payload(self, **kwargs):
            pass
    mock = MockQdrant()
    return mock

@pytest.fixture
def mock_embedding(monkeypatch):
    """Mock embedding function returning fixed-length vectors."""
    import random
    def fake_embed(text, prefix=""):
        random.seed(hash(text))
        return [random.random() for _ in range(768)]
    return fake_embed
```
**Tests to add:** This IS the test infrastructure
**Dependencies:** None

---

### Task 61: Add BM25 Unit Tests
**Priority:** P0-CRITICAL
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** Create `tests/test_bm25.py`
**Current behavior:** Zero tests for BM25.
**Expected behavior:** Tests for tokenization, scoring, RRF fusion.
**Fix description:**
```python
from bm25_search import hybrid_rerank, reciprocal_rank_fusion

def test_tokenize_basic():
    from bm25_search import tokenize
    tokens = tokenize("Hello, World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens

def test_tokenize_cyrillic():
    tokens = tokenize("Москва — столица России")
    assert "москва" in tokens

def test_bm25_scoring_relevance():
    results = [
        {"text": "The quick brown fox jumps over the lazy dog"},
        {"text": "Python programming language"},
        {"text": "Quick fox is fast"},
    ]
    reranked = hybrid_rerank("quick fox", results, bm25_weight=1.0)
    # Results with "quick" and "fox" should score higher
    assert "fox" in reranked[0]["text"].lower()

def test_rrf_equal_length():
    dense = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
    bm25_scores = [0.5, 0.7]
    result = reciprocal_rank_fusion(dense, bm25_scores)
    assert len(result) == 2

def test_rrf_empty():
    result = reciprocal_rank_fusion([], [])
    assert result == []
```
**Tests to add:** This IS the test file
**Dependencies:** Task 3 (fix tokenizer first)

---

### Task 62: Add Commit Pipeline Unit Tests
**Priority:** P0-CRITICAL
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** Create `tests/test_commit.py`
**Fix description:**
Test A-MAC scoring parser, dedup logic, payload construction:
```python
def test_amac_parser_extracts_triplet():
    # Test the regex parser with known good input
    pass

def test_amac_parser_ignores_examples():
    # Verify parser doesn't return prompt example values
    pass

def test_dedup_detects_near_duplicate():
    pass

def test_dedup_allows_different_text():
    pass

def test_payload_has_required_fields():
    # Verify text, source, date, importance, embedding_model, schema_version
    pass

def test_importance_clamped():
    pass

def test_protected_fields_not_overwritten():
    pass
```
**Tests to add:** This IS the test file
**Dependencies:** Tasks 17-20 (validation must be in place)

---

### Task 63: Add Search Pipeline Unit Tests
**Priority:** P0-CRITICAL
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** Create `tests/test_search.py`
**Fix description:**
```python
def test_temporal_decay_reduces_old_scores():
    pass

def test_temporal_decay_preserves_recent():
    pass

def test_multifactor_scoring():
    pass

def test_dedup_removes_same_thread():
    pass

def test_source_tiering_weights():
    pass
```
**Tests to add:** This IS the test file
**Dependencies:** Phase 2 (unified pipeline)

---

### Task 64: Add Entity Extraction Tests
**Priority:** P1-HIGH
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** Create `tests/test_entities.py`
**Fix description:**
```python
def test_extract_known_person():
    pass

def test_extract_known_org():
    pass

def test_no_substring_match():
    # "Al" should not match "algorithm"
    pass

def test_capitalized_phrase_extraction():
    pass

def test_entity_extraction_consistency():
    # Same text returns same entities every time
    pass
```
**Tests to add:** This IS the test file
**Dependencies:** Task 46

---

### Task 65: Add Integration Test (Commit → Search Round-Trip)
**Priority:** P1-HIGH
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** Create `tests/test_integration.py`
**Fix description:**
This test requires running services (Qdrant, Ollama). Skip in CI if services unavailable:
```python
import pytest
import requests

BRAIN_URL = "http://localhost:7777"

@pytest.fixture(autouse=True)
def skip_if_no_server():
    try:
        requests.get(f"{BRAIN_URL}/health", timeout=2)
    except:
        pytest.skip("Memory server not running")

def test_commit_and_search():
    # Commit a unique memory
    text = f"Integration test memory: {time.time()}"
    resp = requests.post(f"{BRAIN_URL}/commit",
        json={"text": text, "source": "test", "importance": 70, "force": True})
    assert resp.status_code == 200
    
    # Search for it
    time.sleep(1)  # Allow indexing
    resp = requests.get(f"{BRAIN_URL}/search", params={"q": text[:50], "limit": 5})
    assert resp.status_code == 200
    results = resp.json().get("results", [])
    assert any(text[:50] in r.get("text", "") for r in results)
```
**Tests to add:** This IS the test file
**Dependencies:** Phases 1-2

---

### Task 66: Create Ground Truth Dataset
**Priority:** P1-HIGH
**Source:** 05-tests-benchmarks.md, opus-05-tests-crossexam.md
**Validated by:** Both agree
**File(s):** Create `benchmarks/ground_truth.jsonl`
**Current behavior:** No benchmark dataset. No way to measure recall quality.
**Expected behavior:** 50+ query-answer pairs for measuring Recall@5, MRR.
**Fix description:**
Extract the 7 test queries from `run_tests()` in hybrid_brain.py (before deleting it) and expand:
```jsonl
{"query": "dad health transplant surgery", "expected_keywords": ["transplant", "Toronto", "lung"]}
{"query": "casino revenue DACH", "expected_keywords": ["revenue", "DACH", "German"]}
{"query": "fertility IVF supplements", "expected_keywords": ["IVF", "Orthomol", "fertility"]}
...
```
Create 50 queries covering different domains (health, business, family, tech, personal).
**Tests to add:** `test_recall_at_5()` — automated benchmark
**Dependencies:** Phase 1 (bugs fixed first)

---

### Task 67: Add Config Loading Tests
**Priority:** P2-MEDIUM
**Source:** Task 30 dependency
**File(s):** Create `tests/test_config.py`
**Fix description:**
```python
def test_config_loads_from_toml():
    from config import load_config
    cfg = load_config("config/rasputin.toml")
    assert cfg["qdrant"]["collection"] == "second_brain"

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    from config import load_config
    cfg = load_config("config/rasputin.toml")
    assert cfg["qdrant"]["collection"] == "test_collection"
```
**Tests to add:** This IS the test file
**Dependencies:** Task 30

---

### Task 68: Add Decay System Tests
**Priority:** P2-MEDIUM
**Source:** 05-tests-benchmarks.md
**File(s):** Create `tests/test_decay.py`
**Fix description:**
```python
def test_decay_uses_correct_collection():
    from memory_decay import COLLECTION
    assert COLLECTION == "second_brain" or "second_brain" in COLLECTION

def test_high_importance_protected():
    # Memory with importance=95, 200 days old → NOT soft-deleted
    pass

def test_low_importance_archived():
    # Memory with importance=20, 100 days old → archived
    pass

def test_temporal_decay_math():
    # Verify Ebbinghaus formula produces expected values
    pass
```
**Tests to add:** This IS the test file
**Dependencies:** Tasks 1, 9

---

---

## Phase 7: CI/CD & Documentation

**Goal:** CI runs real tests with service containers. Documentation is complete and accurate.

**Expected outcome:** Every push runs linting, type checking, unit tests, and (optionally) integration tests. README reflects v3.0 architecture.

---

### Task 69: Upgrade CI with Service Containers
**Priority:** P1-HIGH
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** `.github/workflows/ci.yml`
**Current behavior:** CI only runs smoke tests (parse-only). No Qdrant, FalkorDB, or Ollama.
**Expected behavior:** CI starts Qdrant for integration tests.
**Fix description:**
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333

    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff pytest pytest-cov mypy
      
      - name: Lint
        run: ruff check .
      
      - name: Type check
        run: mypy tools/hybrid_brain.py tools/bm25_search.py --ignore-missing-imports || true
      
      - name: Unit tests
        run: pytest tests/ -v --cov=tools --cov-report=xml -k "not integration"
      
      - name: Integration tests
        run: pytest tests/test_integration.py -v || true
        env:
          QDRANT_URL: http://localhost:6333
```
**Tests to add:** N/A (CI config)
**Dependencies:** Tasks 60-68 (tests must exist)

---

### Task 70: Add Coverage Reporting
**Priority:** P2-MEDIUM
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** `.github/workflows/ci.yml`, `pyproject.toml`
**Current behavior:** No coverage tracking.
**Expected behavior:** Coverage reported on every PR with 60% minimum gate.
**Fix description:**
In `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=tools --cov-report=term-missing"

[tool.coverage.run]
source = ["tools"]
omit = ["tools/memory_autogen.py", "tools/embed_server_gpu1.py"]

[tool.coverage.report]
fail_under = 60
```
**Tests to add:** N/A
**Dependencies:** Tasks 60-68

---

### Task 71: Add mypy Configuration
**Priority:** P2-MEDIUM
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** `pyproject.toml`
**Fix description:**
```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_configs = true
disallow_untyped_defs = false  # Gradual adoption
```
**Tests to add:** N/A
**Dependencies:** Tasks 34, 35

---

### Task 72: Pin Requirements Versions
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree
**File(s):** `requirements.txt`
**Current behavior:** All `>=` constraints allow arbitrary upgrades.
**Expected behavior:** Pinned versions for reproducibility.
**Fix description:**
Run `pip freeze` and pin current working versions. Split into core and optional:
```
# requirements.txt (core)
qdrant-client==1.9.0
redis==5.0.0
requests==2.31.0
python-dotenv==1.0.0

# requirements-extras.txt (optional heavy deps)
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.0.0
```
**Tests to add:** N/A
**Dependencies:** None

---

### Task 73: Update README.md
**Priority:** P1-HIGH
**Source:** 00-master-architecture-review.md
**Validated by:** Master review
**File(s):** `README.md`
**Current behavior:** README may not reflect v3.0 changes.
**Expected behavior:** Complete, accurate documentation of architecture, setup, and usage.
**Fix description:**
Update to include:
1. v3.0 architecture diagram (from master review)
2. Quick start guide (docker compose up + python3 hybrid_brain.py)
3. Configuration reference (rasputin.toml fields)
4. API reference (all endpoints with examples)
5. Development guide (how to run tests, add features)
6. Changelog (what changed from v2.0)
**Tests to add:** N/A
**Dependencies:** All previous phases

---

### Task 74: Add CHANGELOG.md
**Priority:** P2-MEDIUM
**Source:** General best practice
**File(s):** Create `CHANGELOG.md`
**Fix description:**
Document all changes made during this work order, organized by phase.
**Tests to add:** N/A
**Dependencies:** All previous phases

---

### Task 75: Add Benchmark Runner
**Priority:** P2-MEDIUM
**Source:** 05-tests-benchmarks.md
**Validated by:** Both agree
**File(s):** Create `benchmarks/run_benchmark.py`
**Fix description:**
```python
"""Benchmark runner for recall quality and latency."""
import json
import time
import requests

def run_benchmarks(ground_truth_path="benchmarks/ground_truth.jsonl", api_url="http://localhost:7777"):
    queries = [json.loads(line) for line in open(ground_truth_path)]
    
    latencies = []
    recall_at_5 = 0
    mrr_sum = 0
    
    for q in queries:
        start = time.time()
        resp = requests.get(f"{api_url}/search", params={"q": q["query"], "limit": 5})
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        results = resp.json().get("results", [])
        # Check if expected keywords appear in top 5
        for i, r in enumerate(results):
            text = r.get("text", "").lower()
            if any(kw.lower() in text for kw in q["expected_keywords"]):
                recall_at_5 += 1
                mrr_sum += 1 / (i + 1)
                break
    
    print(f"Recall@5: {recall_at_5}/{len(queries)} ({recall_at_5/len(queries)*100:.1f}%)")
    print(f"MRR: {mrr_sum/len(queries):.3f}")
    print(f"Latency p50: {sorted(latencies)[len(latencies)//2]:.0f}ms")
    print(f"Latency p95: {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms")

if __name__ == "__main__":
    run_benchmarks()
```
**Tests to add:** N/A (this IS the benchmark)
**Dependencies:** Task 66 (ground truth dataset)

---

### Task 76: Add Rate Limiting to API
**Priority:** P2-MEDIUM
**Source:** 06-infrastructure-api.md
**Validated by:** Both agree (but Opus noted lower priority for localhost)
**File(s):** `tools/hybrid_brain.py`
**Current behavior:** Zero rate limiting. Any client can flood the API.
**Expected behavior:** Basic per-endpoint rate limiting.
**Fix description:**
```python
import time
from collections import defaultdict

class SimpleRateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.history = defaultdict(list)
    
    def allow(self, key="default"):
        now = time.time()
        self.history[key] = [t for t in self.history[key] if now - t < 60]
        if len(self.history[key]) >= self.calls_per_minute:
            return False
        self.history[key].append(now)
        return True

_rate_limiters = {
    "/commit": SimpleRateLimiter(calls_per_minute=30),
    "/search": SimpleRateLimiter(calls_per_minute=120),
}
```
Check in handler before processing request.
**Tests to add:** `test_rate_limiter_blocks_excess()`
**Dependencies:** None

---

### Task 77: Split requirements.txt into Core and Extras
**Priority:** P3-LOW
**Source:** opus-06-infrastructure-crossexam.md
**Validated by:** Opus only
**File(s):** `requirements.txt`
**Current behavior:** Single requirements.txt includes heavy optional deps (torch, transformers, etc.)
**Expected behavior:** Core deps separate from optional.
**Fix description:**
Create `requirements-core.txt` (qdrant-client, redis, requests, python-dotenv) and `requirements-ml.txt` (torch, transformers, sentence-transformers). Update Dockerfile and quickstart.sh to install core by default.
**Tests to add:** N/A
**Dependencies:** None

---

### Task 78: Add Cron Lock Files
**Priority:** P2-MEDIUM
**Source:** 03-maintenance-lifecycle.md
**Validated by:** Both agree
**File(s):** `tools/memory_decay.py`, `tools/memory_dedup.py`, `tools/fact_extractor.py`, `tools/memory_consolidator_v4.py`
**Current behavior:** No lock files. Concurrent cron runs can cause race conditions. decay might run before consolidation finishes.
**Expected behavior:** Lock file prevents concurrent execution.
**Fix description:**
Add to each maintenance script:
```python
import fcntl

LOCK_FILE = "/tmp/rasputin_decay.lock"

def acquire_lock():
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except IOError:
        print("Another instance is running. Exiting.")
        sys.exit(0)
```
**Tests to add:** N/A
**Dependencies:** None

---

---

## Summary

| Phase | Tasks | Estimated Effort |
|-------|-------|-----------------|
| **Phase 1: Critical Bug Fixes** | Tasks 1-25 | ~20-25 hours |
| **Phase 2: Pipeline Unification** | Tasks 26-29 | ~12-16 hours |
| **Phase 3: Architecture Cleanup** | Tasks 30-45 | ~16-20 hours |
| **Phase 4: Graph Layer Overhaul** | Tasks 46-53 | ~10-14 hours |
| **Phase 5: New Features** | Tasks 54-59 | ~16-20 hours |
| **Phase 6: Test Suite** | Tasks 60-68 | ~16-20 hours |
| **Phase 7: CI/CD & Documentation** | Tasks 69-78 | ~12-16 hours |
| **TOTAL** | **78 tasks** | **~100-130 hours** |

**Priority Distribution:**
- P0-CRITICAL: 12 tasks (fix immediately)
- P1-HIGH: 28 tasks (fix this sprint)
- P2-MEDIUM: 28 tasks (fix this month)
- P3-LOW: 10 tasks (nice to have)

**Git Workflow:**
- After Phase 1: `git commit -m "fix(phase-1): critical bug fixes — decay collection, BM25, fact extractor, concurrency"`
- After Phase 2: `git commit -m "feat(phase-2): unified search pipeline — merge memory_engine into hybrid_brain"`
- After Phase 3: `git commit -m "refactor(phase-3): config consolidation, dead code removal, type hints"`
- After Phase 4: `git commit -m "fix(phase-4): graph layer — unified NER, consistent relationships, parameterized queries"`
- After Phase 5: `git commit -m "feat(phase-5): contradiction detection, importance recalc, feedback loop"`
- After Phase 6: `git commit -m "test(phase-6): comprehensive test suite — unit, integration, benchmark"`
- After Phase 7: `git commit -m "ci(phase-7): CI with service containers, coverage, documentation"`

Push after each phase: `git push origin main`

---

*Work order generated from 13 audit reports by 13 AI agents.*
*Ready for execution by OpenCode Zen (Claude Opus).*
