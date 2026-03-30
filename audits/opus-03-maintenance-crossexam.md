# Opus Cross-Examination: Memory Maintenance (Decay, Dedup, Consolidation)

**Cross-examiner:** Opus 4.6 (second-pass review)  
**Original audit:** Qwen 122B — `03-maintenance-lifecycle.md`  
**Date:** 2026-03-30  
**Scope:** `rasputin-memory/tools/` — memory_decay.py, memory_dedup.py, memory_consolidate.py, memory_consolidator_v4.py, consolidate_second_brain.py, hybrid_brain.py

---

## 1. Confirmed Findings

These 122B findings are accurate and correctly characterized:

### ✅ Collection Name Mismatch (HIGH — confirmed)
`memory_decay.py` hardcodes `COLLECTION = "memories_v2"` (line 32) while everything else uses `"second_brain"`. This means **the decay engine operates on a completely different collection** than the rest of the system. If `memories_v2` is empty or doesn't exist, decay silently does nothing. If it exists with stale data, it archives the wrong things. This is arguably **CRITICAL**, not just HIGH.

### ✅ Non-Atomic Archive-Then-Delete (HIGH — confirmed)
The upsert-then-delete pattern in `memory_decay.py` lines 273-293 and 330-355 is indeed non-atomic. Crash between the two = data loss or duplication. Real bug.

### ✅ No Checkpoint in Consolidation Pipeline (HIGH — confirmed)
`memory_consolidate.py` runs 5 LLM passes sequentially with no intermediate persistence. A timeout at pass 4 loses all work. Confirmed by reading the code.

### ✅ Multiple Dedup Tools (HIGH — confirmed)
At least `memory_dedup.py`, `check_dedup.py`, `dedup_scan.py`, `librarian_dedup.py` exist with different algorithms and thresholds. Fragmentation risk is real.

### ✅ Access Tracking Broken (HIGH — confirmed)
`_update_access_tracking()` in `hybrid_brain.py` (line 1111) uses `text[:200]` matching via scroll to find points, which is unreliable. The spaced-repetition feedback loop is effectively dead.

### ✅ Temporal Decay Formula Mislabeled (MEDIUM — confirmed)
Comment says "power-law" but formula is exponential decay (`e^(-t/S)`). Functionally fine; naming is wrong.

---

## 2. Missed Issues (NEW — 122B didn't catch these)

### 🔴 CRITICAL: Decay Engine Targets Wrong Collection Entirely

**File:** `memory_decay.py`, line 32  
**Issue:** `COLLECTION = "memories_v2"` is hardcoded, while the actual live collection is `"second_brain"`. Every other tool (`hybrid_brain.py`, `memory_dedup.py`, `memory_engine.py`, `smart_memory_query.py`) uses `"second_brain"`.

The 122B auditor flagged this as a "collection name mismatch" (HIGH) but **completely underestimated the impact**. This isn't a consistency issue — it means **the entire decay engine is a no-op against production data**. It's scanning and archiving from a collection that likely has zero or stale points while `second_brain` grows unbounded.

**Severity:** CRITICAL — decay is completely non-functional against the live collection.  
**Fix:** Change line 32 to `COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")`.

### 🔴 CRITICAL: consolidator_v4 Bypasses Quality Gates and Commits Directly to Qdrant

**File:** `memory_consolidator_v4.py`, `commit_to_brain()` function (around line 180)  
**Issue:** The v4 consolidator bypasses the normal `hybrid_brain.py` commit pipeline and writes directly to Qdrant via raw HTTP PUT. The comment even says: *"bypasses A-MAC quality gate. These facts are already LLM-curated so double-scoring wastes GPU."*

Problems:
1. **No duplicate checking** — commits facts without checking if they already exist in Qdrant
2. **No importance scoring** — payload has no `importance` field, so decay engine will treat all v4 facts as importance=50 (default)
3. **No `last_accessed` field** — decay engine's `get_last_accessed()` will fall back to `date` field, but v4 uses `timestamp` instead of `date`
4. **Hash-based dedup is MD5 of first 200 chars** — trivially collides on short facts, and doesn't catch semantic duplicates at all

**Severity:** CRITICAL — this is a data quality bypass that can flood the brain with unscored, undeduplicated facts.  
**Fix:** At minimum add `importance`, `date` (not just `timestamp`), and a vector-similarity dedup check before commit.

### 🟠 HIGH: Stale Read in Access Tracking Increment

**File:** `hybrid_brain.py`, line 1147  
**Issue:** `current_count = r.get("retrieval_count", 0)` reads from the *search result dict* (which was fetched moments ago), not from the current Qdrant payload. If two concurrent searches return the same memory, both read `retrieval_count=5`, both write `6`. The count should be `7`. This is a classic lost-update race condition.

The 122B auditor mentioned race conditions generally but didn't identify this specific stale-read pattern.

**Severity:** HIGH — systematically undercounts retrievals, degrading spaced-repetition accuracy.  
**Fix:** Read current payload from Qdrant inside the update thread, or use Qdrant's payload operations if they support atomic increment.

### 🟠 HIGH: consolidate_second_brain.py Deletes Source Collections After Unverified Migration

**File:** `consolidate_second_brain.py`, `_verify_migration()` method  
**Issue:** The "verification" just does a scroll with limit=1 and checks if status code is 200. It doesn't verify the count matches. Then `_delete_collection()` immediately deletes the source. If the migration was partial (network timeout, embedding failures), you lose data permanently.

**Severity:** HIGH — one-way data loss on partial migration.  
**Fix:** Compare migrated count against source count before deleting.

### 🟠 HIGH: consolidator_v4 Embedding Validation Is Incomplete

**File:** `memory_consolidator_v4.py`, `commit_to_brain()` function  
**Issue:** The code checks `if mag < 0.1` to reject garbage embeddings, but nomic-embed-text produces normalized vectors with magnitude ≈ 1.0. A magnitude of 0.5 would be severely corrupted but would pass this check. The threshold should be much higher (e.g., 0.8).

Additionally, the embedding endpoint used is `http://localhost:11434/api/embeddings` (singular) while the dedup tool uses `http://localhost:11434/api/embed` — both work but it's inconsistent and fragile.

**Severity:** HIGH — corrupted embeddings can enter the brain and pollute search results.  
**Fix:** Raise magnitude threshold to 0.8, normalize endpoint URLs.

### 🟡 MEDIUM: memory_dedup.py Checkpoint Doesn't Save processed_ids

**File:** `memory_dedup.py`, line 217  
**Code:** `# Don't save full processed_ids to checkpoint (too large), just the offset`  
**Issue:** On resume, the `processed_set` is empty. If a point was processed but its dupes span across batch boundaries, the resume run won't know which points were already evaluated. This can lead to re-processing and potentially deleting the "keeper" from a previous cluster.

**Severity:** MEDIUM — resume after crash can produce inconsistent results.  
**Fix:** Save processed_ids to a separate file (one ID per line) or use a bloom filter.

### 🟡 MEDIUM: Soft-Delete Applies to ALL Memories >180 Days Regardless of Importance

**File:** `memory_decay.py`, lines 206-220  
**Issue:** The 122B auditor flagged this but misdescribed the code. Looking at the actual scan logic:

```python
if days_since_access >= SOFT_DELETE_DAYS:  # 180 days
    stats["soft_delete_candidates"] += 1
    soft_delete_candidates.append(...)
elif days_since_access >= ARCHIVE_DAYS and importance < LOW_IMPORTANCE_THRESHOLD:
    stats["archive_candidates"] += 1
```

The `elif` means: memories >180 days go to soft-delete **regardless of importance**. Only the 90-180 day range checks importance. So a memory with `importance=95` that hasn't been accessed in 181 days gets soft-deleted. The 122B auditor was right about the *bug* but the code structure is slightly different than they quoted.

**Severity:** Should be CRITICAL (122B also rated it CRITICAL, but misdescribed the mechanism).

### 🟡 MEDIUM: consolidator_v4 Has No Global Dedup Against Existing Brain Content

**File:** `memory_consolidator_v4.py`  
**Issue:** Hash-based dedup (`seen_hashes`) only deduplicates within the current v4 run. It loads hashes from `extracted-v4.jsonl` but never checks against existing Qdrant content. If the same fact was committed by `hybrid_brain.py`, `fact_extractor.py`, or a previous v4 run that was restarted with a fresh output file, it creates duplicates.

**Severity:** MEDIUM — gradual brain pollution over multiple runs.

### 🔵 LOW: consolidate_second_brain.py Uses Unstable Point IDs

**File:** `consolidate_second_brain.py`, `_process_batch()`  
**Issue:** Reuses original point IDs from source collections when upserting into `second_brain`. If two source collections happen to have the same point ID (possible with UUID collision or integer IDs), the second upsert silently overwrites the first.

**Severity:** LOW — unlikely but possible with integer-ID collections.

---

## 3. Corrections (Where 122B Was Wrong or Inaccurate)

### ❌ "Double-Penalty Bug in Importance Scoring" — Mischaracterized

The 122B audit claims there's a "double penalty" where `compute_importance_score` is called but the "original static importance field" is used instead. This is **wrong**. Reading the actual code:

```python
# Line 224 (memory_decay.py)
importance = compute_importance_score(payload)

# Line 229
elif days_since_access >= ARCHIVE_DAYS and importance < LOW_IMPORTANCE_THRESHOLD:
```

The variable `importance` IS the computed score. There's no "double penalty." The decay engine correctly uses the dynamic score for its threshold check. The 122B auditor confused themselves — perhaps they expected the computed score to be written back to the payload, but it doesn't need to be for the decay decision to be consistent.

**Verdict:** FALSE POSITIVE. Remove from CRITICAL. The function works correctly for its purpose within `memory_decay.py`.

### ❌ "O(n²) Scaling Concerns" — Overstated for memory_dedup.py

The 122B auditor lumped `memory_dedup.py` with the old `librarian_dedup.py` and claimed O(n²). But `memory_dedup.py` uses **Qdrant vector search** per point (line 127: `query_points`), which is O(n log n) via HNSW. The nested-loop O(n²) only applies to the legacy `librarian_dedup.py` and `dedup_scan.py` which use `SequenceMatcher`. The modern dedup tool already does what the auditor recommended.

**Verdict:** Partially wrong. The production dedup tool (`memory_dedup.py`) scales well. The legacy tools are indeed O(n²) but should just be deleted, not rewritten.

### ❌ Inflated File Count

The 122B audit lists 14 "files audited" including `proactive_memory.py`, `memory_autocommit.py`, `librarian_dedup_delete.py` — but then says things like "(not read, but inferred from pattern)" for some. Several of the specific code quotes and line numbers don't match the actual files in `rasputin-memory/tools/`. The auditor was likely reading from a different directory or hallucinating line numbers in several places.

**Verdict:** Line number references should be treated as approximate, not exact.

---

## 4. Deeper Analysis

### Collection Mismatch — Full Impact Assessment

The `memories_v2` vs `second_brain` split is more damaging than either audit initially conveyed:

| Tool | Collection | Effect |
|------|-----------|--------|
| `hybrid_brain.py` | `second_brain` | All reads/writes go here |
| `memory_engine.py` | `second_brain` | API server uses this |
| `memory_dedup.py` | `second_brain` | Dedup targets correct collection |
| **`memory_decay.py`** | **`memories_v2`** | **Decay is completely blind to production data** |
| `consolidator_v4.py` | `second_brain` (via raw HTTP) | Writes correctly but bypasses quality |
| `consolidate_second_brain.py` | `second_brain` | Migration tool, one-time use |

**Net effect:** The brain grows indefinitely with zero pruning. Old, low-importance memories accumulate forever, degrading search quality over time. This is the single most impactful bug in the maintenance system.

### Access Tracking — Deeper Failure Analysis

The `_update_access_tracking` function (hybrid_brain.py:1111) has **three compounding failures**:

1. **Text matching is truncated** — `text[:200]` match can return wrong point if two memories share a 200-char prefix
2. **Stale read for increment** — reads count from search results, not Qdrant (concurrent lost-updates)
3. **Cap at 10 updates** — if a query returns 15 results, the last 5 never get tracking updates
4. **Daemon thread** — if the main process exits quickly, daemon threads are killed mid-write

The result: `retrieval_count` is systematically inaccurate, which means the temporal decay's spaced-repetition boost (`effective_half_life = base_half_life * (1 + 0.1 * min(retrieval_count, 20))`) barely activates. Memories that are frequently searched don't get protected from decay (if decay were even running against the right collection).

### consolidator_v4 — Architecture Smell

This tool runs 8 parallel LLM workers extracting facts and committing directly to Qdrant. It's essentially a parallel ETL pipeline that:
- Reads from session logs (raw JSONL)
- Extracts facts via LLM
- Deduplicates via MD5 hash (within-run only)
- Commits directly to Qdrant bypassing all quality gates

It's well-engineered for throughput (ThreadPoolExecutor, checkpoint/resume, rate limiting), but architecturally it's a second data ingestion path that doesn't share any validation logic with `hybrid_brain.py`. Over time this creates two classes of memories: properly scored ones (via hybrid_brain) and raw LLM extractions (via consolidator_v4) with no importance scoring, no graph connections, and no semantic dedup.

---

## 5. Revised Grade

### Original 122B Grade: Not explicitly stated, but implied ~C+ (many issues, some critical)

### My Revised Assessment: **D+**

**Reasoning:**

The 122B audit identified real problems but missed the most damaging one: **the decay engine targets a completely different collection than production**. This single bug means the entire maintenance lifecycle is non-functional — memories are never pruned, never archived, never decayed. Combined with the consolidator_v4 bypass creating unscored data, and broken access tracking preventing spaced-repetition, the maintenance system is essentially decorative.

**What works:**
- `memory_dedup.py` is well-designed (checkpoint/resume, vector-based dedup, quality scoring, batch operations)
- `memory_consolidate.py` has a sound 5-pass architecture even if it lacks checkpointing
- Dry-run defaults everywhere — safe by design

**What's broken:**
- Decay targets wrong collection (CRITICAL — complete functional failure)
- consolidator_v4 bypasses quality gates (CRITICAL — data pollution)
- Access tracking lost-update race (HIGH — spaced repetition broken)
- No importance floor on soft-delete (CRITICAL — can delete critical memories)
- Non-atomic archive-then-delete (HIGH — crash = data loss)

**Priority fix order:**
1. Fix `memory_decay.py` COLLECTION to `"second_brain"` (5 minutes, immediate impact)
2. Add importance floor to soft-delete path (30 minutes)
3. Add `date` and `importance` fields to consolidator_v4 commits (1 hour)
4. Fix access tracking to read current count from Qdrant (2 hours)
5. Add count verification to consolidate_second_brain.py before source deletion (1 hour)

---

*Report generated: 2026-03-30 by Opus 4.6 cross-examination*
