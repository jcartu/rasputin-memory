# Consistency Overhaul — Execution Plan

## Phase A: Unify scoring (the foundation everything depends on)

### A1. Create `tools/pipeline/scoring_constants.py`
One source of truth for source importance across all lifecycle tools:
```python
SOURCE_IMPORTANCE = {
    "conversation": 0.95,  # direct user interaction
    "chatgpt": 0.90,
    "perplexity": 0.90,
    "email": 0.90,
    "telegram": 0.75,
    "whatsapp": 0.70,
    "social_intel": 0.65,
    "consolidator": 0.50,
    "auto-extract": 0.40,
    "fact_extractor": 0.40,
    "web_page": 0.35,
}
```
Delete `pipeline/source_tiering.py` — replace all imports with `scoring_constants`.

### A2. Update memory_decay.py source scoring
Replace the hardcoded `"conversation": 12` dict with:
`importance = int(SOURCE_IMPORTANCE.get(source, 0.5) * 15)`
Same scale, derived from one table.

### A3. Update memory_dedup.py source scoring
Replace `"conversation": 8` dict with:
`priority = int(SOURCE_IMPORTANCE.get(source, 0.5) * 10)`

### A4. Create `tools/pipeline/dateparse.py`
One `parse_date(s: str) -> datetime | None` function. Delete the 3 copies in:
- brain/scoring.py `_parse_date()`
- importance_recalculator.py `_parse_iso()`
- memory_decay.py (inline fromisoformat calls)

### A5. Update brain/commit.py
Import `SOURCE_IMPORTANCE` from `scoring_constants` instead of `source_tiering`.

---

## Phase B: Extract shared utilities

### B1. Create `tools/pipeline/locking.py`
```python
def acquire_lock(name: str) -> int:
    lock_path = f"/tmp/rasputin_{name}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd
```
Replace the 4 copies in memory_decay, memory_dedup, memory_consolidator_v4, fact_extractor.

### B2. Create `tools/pipeline/qdrant_batch.py`
```python
def scroll_all(qdrant_client, collection, batch_size=100, scroll_filter=None, with_payload=True):
    offset = None
    while True:
        points, offset = qdrant_client.scroll(...)
        if not points:
            break
        yield from points
        if offset is None:
            break
```
Replace the 5+ reimplementations.

### B3. Create `tools/pipeline/checkpoint.py`
Unified checkpoint/resume with one JSON format. Used by dedup, consolidator, importance_recalculator.

---

## Phase C: Fix silent graph failures

### C1. commit.py — make graph failure visible
Change from `"ok": True` with buried `"graph": {"written": False}` to:
- `"ok": True, "warnings": ["graph_write_failed: {error}"]`
- Log at WARNING level (not swallowed)

### C2. Define degradation policy in brain/_state.py
```python
DEGRADATION_POLICY = {
    "graph": "warn",       # log + return ok with warning
    "reranker": "silent",  # fall back to unranked
    "embedding": "fail",   # hard error, commit fails
}
```

---

## Phase D: Dead code triage

### D1. Move one-off scripts to `scripts/`
- `consolidate_second_brain.py` → `scripts/consolidate_second_brain.py`
- `migrate_graph_edges.py` → `scripts/migrate_graph_edges.py`
- `backfill_schema.py` → `scripts/backfill_schema.py`

### D2. Delete truly dead code
- `_ = entities` in brain/search.py
- `_ = _state` in brain/scoring.py (if still there after Phase A)

### D3. Document which tools are optional
Add a comment block to the top of tools that are cron/batch jobs vs always-running services.

---

## Phase E: Fix fragile double-import

### E1. Create `tools/pipeline/_imports.py`
```python
def safe_import(primary: str, fallback: str):
    try:
        return importlib.import_module(primary)
    except ModuleNotFoundError:
        return importlib.import_module(fallback)
```
Replace all 8+ try/except import blocks with one-liners.

---

## Phase F: Transaction safety

### F1. memory_decay.py archive
Add existence check before re-upserting in `recover_pending_archives()`:
- Query archive collection for the point ID
- If already there, just delete from main
- If not, re-archive then delete

---

## Phase G: Structured logging

### G1. Create logging config in brain/_state.py
```python
import logging, json, uuid

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "request_id": getattr(record, "request_id", None),
        })
```

### G2. Add request_id to HybridHandler
Generate UUID per request in do_GET/do_POST, attach to log context.

---

## Phase H: Cleanup

### H1. Remove redundant Redis container from docker-compose
FalkorDB is Redis-compatible. Document that FalkorDB serves both graph and cache.

### H2. Fix score ordering
Move multifactor scoring BEFORE RRF fusion in search pipeline.

### H3. Remove `_ = entities` and `_ = _state` dead references.
