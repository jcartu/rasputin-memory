# Opus Cross-Examination — Retrieval Pipeline Audit

**Date:** 2026-03-30  
**Cross-Examiner:** Opus (second-pass review)  
**First Auditor:** Qwen 122B  
**Scope:** Verification of 122B findings + new issue discovery

---

## Confirmed Findings

The 122B audit correctly identified these issues. I agree with the characterization:

1. **✅ BM25 tokenizer strips non-ASCII** (`bm25_search.py:17`) — Correct. The regex `[a-zA-Z0-9]+` destroys Cyrillic. Severity CRITICAL is appropriate given Josh's Moscow context.

2. **✅ BM25 IDF computed per-batch, not corpus-wide** (`bm25_search.py:27-41`) — Correct. IDF is computed only over the ~50 candidate documents, not the 134K corpus. This makes BM25 scores noisy but not catastrophic since it's fused via RRF.

3. **✅ RRF implementation is correct** — Verified. Standard k=60, mathematically sound.

4. **✅ Recency double-counted** — Temporal decay in `apply_temporal_decay()` and then `recency_bonus` in `apply_multifactor_scoring()` both penalize old results. Confirmed.

5. **✅ Importance threshold cliff (79→80)** — Step function at lines 573-579 of `hybrid_brain.py`. The 6x jump from 60-day to 365-day half-life is real.

6. **✅ Reranker fallback is graceful** — Both `memory_engine.py:rerank()` and `hybrid_brain.py:neural_rerank()` fall back to original ordering on failure. Correct.

7. **✅ Embedding model version mismatch** — `embed_server_gpu1.py` loads `nomic-embed-text-v1.5` on port 8003. Main pipeline uses Ollama `nomic-embed-text` (v1) on port 11434. Different models, different vector spaces.

---

## Missed Issues (122B Didn't Catch)

### 🔴 CRITICAL: `BM25_AVAILABLE` is undefined — NameError at runtime

**File:** `hybrid_brain.py` lines 1046, 1388  
**Problem:** The variable `BM25_AVAILABLE` is used in two places but **never defined anywhere in the file**. Line 30 imports `bm25_rerank` successfully, but nobody sets `BM25_AVAILABLE = True`. This means:

- Line 1046: `if BM25_AVAILABLE and qdrant_results:` → **NameError** every search
- Line 1388: health endpoint also crashes

**Impact:** If this code path is actually hit, BM25 reranking silently fails (or crashes the entire search). If it ISN'T hit, then BM25 hybrid search has never worked in `hybrid_brain.py` at all — which means the entire BM25 layer the 122B auditor spent multiple sections analyzing is **dead code** in the main API server.

**Fix:** Add `BM25_AVAILABLE = True` after the import on line 30, or wrap in try/except like `memory_engine.py` does.  
**Effort:** 5 minutes. But the implication is enormous — BM25 may never have been active.

---

### 🟠 HIGH: Access tracking uses stale `retrieval_count` from search results

**File:** `hybrid_brain.py` lines 1140-1154  
**Problem:** The `_do_update()` background thread reads `retrieval_count` from the search result dict (`r.get("retrieval_count", 0)`), which was fetched at query time. It then writes `current_count + 1`. But by the time the background thread runs, the actual count in Qdrant may have been incremented by another concurrent search. This is worse than just a race condition — it's a **guaranteed overwrite** that resets the count to a stale value + 1.

**Example:** Count is 5. Two searches return count=5. Both write count=6. Real count should be 7.

**Fix:** Use Qdrant's scroll to fetch fresh count before writing, or implement a proper atomic increment.  
**Effort:** 2-3 hours.

---

### 🟠 HIGH: Access tracking scroll uses exact text match — will miss most results

**File:** `hybrid_brain.py` lines 1135-1142  
**Problem:** `qdrant.scroll()` with `MatchValue(value=text[:200])` does an **exact string match** on the first 200 characters. But `text` in the result dict was already truncated during retrieval (e.g., `mtext[:500]` for graph results, `text[:1000]` for reranker passages). The stored payload may have up to 4000 chars. If `text[:200]` doesn't exactly match the stored `text[:200]` (whitespace, encoding differences), the scroll finds nothing and access tracking silently fails.

**Impact:** Retrieval counts are likely severely under-counted, which breaks the spaced-repetition half-life extension.  
**Fix:** Preserve point IDs through the pipeline instead of re-searching by text.  
**Effort:** 4-6 hours (refactor to carry IDs).

---

### 🟡 MEDIUM: `memory_engine.py` and `hybrid_brain.py` are two separate, divergent retrieval pipelines

**File:** `memory_engine.py` (entire file) vs `hybrid_brain.py` (entire file)  
**Problem:** The 122B audit conflated these as one pipeline. They are actually **two completely separate implementations**:

- `memory_engine.py`: CLI tool + used by hook scripts. Has its own `recall()`, `rerank()`, `batch_embed()`, `expand_queries()`, `deduplicate()`. Talks to Qdrant REST API directly. No temporal decay. No multifactor scoring. No FalkorDB.
- `hybrid_brain.py`: HTTP server on port 7777. Has `hybrid_search()` with Qdrant client, FalkorDB graph, temporal decay, multifactor scoring, BM25 (broken — see above), neural reranking.

The 122B audit's line number references jump between these files as if they're the same system. They have **different scoring, different dedup, different query expansion, different thresholds**.

**Impact:** Depending on which path is called (CLI vs HTTP API), results will differ significantly. The hook (`openclaw-mem`) likely calls the HTTP API (port 7777), while manual `python3 memory_engine.py recall` uses the standalone pipeline.  
**Fix:** Deprecate one. The HTTP server (`hybrid_brain.py`) is clearly more capable.  
**Effort:** 8-16 hours to unify.

---

### 🟡 MEDIUM: `om_lookup()` has same ASCII-only regex as BM25

**File:** `memory_engine.py` line 91  
**Problem:** `re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())` — same ASCII-only pattern. Observational Memory keyword matching is broken for Russian text.  
**Fix:** Use `r'\b\w{3,}\b'` with `re.UNICODE`.  
**Effort:** 5 minutes.

---

### 🟡 MEDIUM: `graph_traverse()` in `memory_engine.py` also has ASCII-only regex

**File:** `memory_engine.py` line 186  
**Problem:** `re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)` — strips all non-Latin characters from graph traversal queries.  
**Fix:** Same Unicode regex fix.  
**Effort:** 5 minutes.

---

### 🟡 MEDIUM: `commit()` in `memory_engine.py` uses Python `hash()` for point IDs

**File:** `memory_engine.py` line 376  
**Problem:** `point_id = abs(hash(text + str(datetime.now()))) % (2**63)` — Python's `hash()` is randomized per process (PYTHONHASHSEED). Two processes committing the same text get different IDs, bypassing dedup. Also, `hash()` output isn't stable across Python versions.

**Compare with:** `hybrid_brain.py` line 473 uses `hashlib.md5` which is deterministic. But it also includes `time.time()` making it unique per call anyway — so dedup relies on the separate `check_duplicate()` cosine check, not ID collision.

**Impact:** Minor — dedup is handled by cosine similarity check in `hybrid_brain.py`. But the `memory_engine.py` commit path has NO dedup check at all.  
**Fix:** Add dedup check to `memory_engine.py:commit()` or route all commits through the HTTP API.  
**Effort:** 2-3 hours.

---

### 🟡 MEDIUM: Reranker max_length=512 tokens truncates long memories

**File:** `reranker_server.py` line 84  
**Problem:** `tokenizer(pairs, max_length=512, truncation=True)` — query + passage together are capped at 512 tokens. For a 50-word query and a 1000-char passage, the passage gets heavily truncated. The reranker only sees the beginning of each memory.

**Impact:** Memories where the relevant content is in the middle or end will be underscored.  
**Fix:** Increase to 1024 (bge-reranker-v2-m3 supports up to 8192 tokens), or extract a more relevant snippet before reranking.  
**Effort:** 1 hour (just change the parameter), but test VRAM impact.

---

### 🔵 LOW: `commit_memory()` doesn't check embedding magnitude for document prefix

**File:** `hybrid_brain.py` lines 455-461  
**Problem:** The garbage vector check (`magnitude < 0.1`) is good, but `nomic-embed-text` with `search_document:` prefix produces normalized embeddings (magnitude ≈ 1.0). The 0.1 threshold only catches completely broken embeddings. A partially broken embedding (magnitude 0.3) would pass but produce garbage search results.  
**Fix:** Raise threshold to 0.5 or check against expected range.  
**Effort:** 15 minutes.

---

### 🔵 LOW: `format_recall()` hardcodes "761K total" count

**File:** `memory_engine.py` line 340  
**Problem:** `"Found {len(results)} relevant memories from second brain (761K total)"` — hardcoded collection size. Actual size is ~134K per the audit. This is cosmetic but misleading.  
**Fix:** Query actual count or remove.  
**Effort:** 5 minutes.

---

## Corrections (Where 122B Was Wrong)

### 1. "No adaptive k selection" rated 🟠 HIGH — should be 🟡 MEDIUM

The 122B audit says `top_k=10` is fixed and this is HIGH severity. But looking at the actual code in `memory_engine.py`, it searches with `top_k=10` across 5-12 expanded queries, then merges into `all_results` dict (deduped by ID, keeping max score). This effectively gives 50-120 candidates before dedup, then takes top 50 for BM25+reranking. The system already has adaptive behavior through multi-query expansion. Fixed k per-query is fine when you're running 12 queries.

### 2. "Reranker called on too many candidates" rated 🟠 HIGH — overstated

The 122B says reranking 50+ candidates is expensive. But looking at `reranker_server.py`, the cross-encoder processes all pairs in a **single batched forward pass** (line 80-86: `tokenizer(pairs, padding=True, ...)`). This is GPU-parallelized, not O(n) API calls. 50 candidates at 512 tokens each = ~25K tokens, which is a single GPU batch on any modern card. The 30-60ms latency is already very fast. This is 🔵 LOW at best.

### 3. "Reranker called on potentially empty results" — not a real issue

The 122B audit flagged `hybrid_brain.py` line 286 for calling reranker on empty results. But `neural_rerank()` has `if not results: return results` at line 126. And the call site in `hybrid_search()` checks `if reranker_up and all_candidates:`. There are two guards. This is a non-issue.

### 4. Line number references are often wrong

The 122B audit references specific line numbers that don't match the actual files. For example:
- "lines 271-291" for recall thresholds — actual thresholds are around lines 275-310
- "lines 431-476" for multifactor scoring — actual function is at lines 600-650 in `hybrid_brain.py`
- "lines 539-572" for access tracking — actual function starts at line 1110

This suggests the 122B auditor was working from memory or a different version of the code, not reading the actual files carefully.

### 5. "Memory footprint at 1M" rated 🔴 CRITICAL — premature

The system has 134K vectors. 1M is 7.5x current scale. Rating a theoretical future scaling issue as CRITICAL alongside an actual runtime bug (BM25 tokenizer) that affects every search today is poor severity calibration. This should be 🟡 MEDIUM (plan for later).

---

## Deeper Analysis

### The BM25_AVAILABLE Bug Changes Everything

The 122B audit spent significant effort analyzing BM25 scoring quality, IDF computation, and RRF fusion in the context of `hybrid_brain.py`. But `BM25_AVAILABLE` is undefined, meaning **BM25 has likely never executed in the HTTP API server**. The import succeeds (line 30), but the guard variable was never set. 

This means:
- All searches through port 7777 skip BM25 entirely
- RRF fusion never happens in the main API
- The only place BM25 works is in `memory_engine.py` (CLI), which has `HAS_BM25 = True` set after import

The 122B auditor analyzed BM25 as if it were active. It isn't. The entire section on BM25 IDF quality is about dead code in the main server.

### Dual Pipeline Architecture is the Real Architectural Issue

The 122B audit missed that there are TWO retrieval pipelines. This is the biggest architectural problem:

| Feature | `memory_engine.py` | `hybrid_brain.py` |
|---------|--------------------|--------------------|
| BM25 | ✅ Works (HAS_BM25) | ❌ Broken (BM25_AVAILABLE undefined) |
| FalkorDB graph | ❌ JSON file only | ✅ Full Cypher traversal |
| Temporal decay | ❌ None | ✅ Ebbinghaus power-law |
| Multifactor scoring | ❌ None | ✅ Full formula |
| Neural reranking | ✅ Works | ✅ Works |
| Query expansion | ✅ Full (12 angles) | ❌ None (single query) |
| Dedup | ✅ Thread/hash based | ❌ None |
| Tiered source search | ✅ Gold/Silver/Bronze | ❌ Single filter |

Neither pipeline has the complete feature set. The ideal would combine `memory_engine.py`'s query expansion + tiered search with `hybrid_brain.py`'s graph + decay + multifactor scoring.

### Access Tracking is Effectively Broken

The access tracking in `hybrid_brain.py` has three compounding issues:
1. Background thread with no point IDs (must re-search by exact text match)
2. Exact text match on truncated text (likely fails often)
3. Stale count overwrite (race condition)

Combined, this means retrieval counts are severely under-counted, which means spaced-repetition half-life extension barely works. Memories that should become more stable through repeated access still decay at their base rate.

---

## Revised Grade

**Overall: B (80/100)** (down from 122B's B+ 85/100)

| Dimension | 122B Grade | My Grade | Reason |
|-----------|-----------|----------|--------|
| Architecture | A (95) | A- (88) | Dual pipeline is a real problem |
| Implementation | B (80) | C+ (75) | BM25_AVAILABLE bug = dead feature, broken access tracking |
| Performance | A- (90) | A- (90) | Agree — good for current scale |
| SOTA comparison | A- (88) | B+ (85) | Without BM25 working in the API server, hybrid search is less hybrid than claimed |

**Key delta from 122B:** The BM25_AVAILABLE bug means the main API server's "hybrid search" is really just vector + graph + reranking. The claimed 7-layer pipeline is actually 5 layers in production. This is still good, but the 122B audit overestimated what's actually running.

---

## Priority Fixes (Revised)

### Immediate (< 1 hour)
1. **Define `BM25_AVAILABLE = True`** after import in `hybrid_brain.py` — enables the entire BM25+RRF layer
2. **Fix all ASCII-only regexes** — `bm25_search.py:17`, `memory_engine.py:91`, `memory_engine.py:186`
3. **Fix hardcoded "761K total"** in `format_recall()`

### This Week
4. **Carry point IDs through the pipeline** so access tracking actually works
5. **Increase reranker max_length** from 512 to 1024
6. **Unify or deprecate** `memory_engine.py` standalone pipeline

### This Month
7. **Smooth the importance cliff** (linear half-life scaling)
8. **Remove recency double-counting**
9. **Pre-compute corpus-wide BM25 IDF**

---

**Report generated:** 2026-03-30T21:45 MSK  
**Cross-examiner:** Opus (claude-opus-4-6)
