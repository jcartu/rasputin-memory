# Operations Guide

Day-to-day maintenance procedures for the RASPUTIN Memory System.

## Deduplication Pipeline

Over time, similar memories accumulate in Qdrant. The dedup pipeline identifies and removes near-duplicates using vector cosine similarity.

### How It Works

1. **Scan** — Iterates through all vectors in the collection in batches
2. **Compare** — Computes cosine similarity between each vector and its nearest neighbors
3. **Cluster** — Groups vectors above the similarity threshold as duplicates
4. **Select** — Keeps the best vector per cluster (highest quality score, most metadata, most recent)
5. **Delete** — Removes the rest

### Default Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Similarity threshold | 0.92 | Cosine similarity above which two vectors are considered duplicates |
| Batch size | 1000 | Vectors processed per batch |
| Checkpoint | Auto | Progress saved to `.dedup_checkpoint.json` for resume |

### Running Dedup

**Always dry-run first:**

```bash
python3 tools/memory_dedup.py --threshold 0.95 --dry-run
```

This reports how many duplicates would be removed without deleting anything.

**Execute (actually delete):**

```bash
python3 tools/memory_dedup.py --threshold 0.95 --execute
```

**Two-pass approach (recommended for first run):**

```bash
# Pass 1: Strict threshold — catch obvious duplicates
python3 tools/memory_dedup.py --threshold 0.95 --execute

# Pass 2: Slightly looser — catch near-duplicates
python3 tools/memory_dedup.py --threshold 0.92 --execute
```

### Expected Results

- First run on a mature collection: typically removes **15–20%** of vectors
- Subsequent monthly runs: 2–5% removal
- Log output is written to `.dedup_log.jsonl`

### Resume After Interruption

```bash
python3 tools/memory_dedup.py --resume --execute
```

Reads from `.dedup_checkpoint.json` and continues where it left off.

## Memory Decay

Applies Ebbinghaus forgetting curve to memory scores, gradually reducing the weight of old, unreinforced memories.

```bash
python3 tools/memory_decay.py
```

- **Half-life:** 30 days (configurable via `TEMPORAL_HALF_LIFE_DAYS`)
- Memories that are re-accessed or re-committed get their decay timer reset
- Run weekly via cron (see [CRON_JOBS.md](CRON_JOBS.md))

## Health Check

Verifies all system components are operational:

```bash
python3 tools/memory_health_check.py
```

Checks: Qdrant connectivity, FalkorDB graph, Ollama embedding model, reranker (if configured).

## Backup

### Qdrant Snapshots

```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/second_brain/snapshots

# List snapshots
curl http://localhost:6333/collections/second_brain/snapshots

# Download a snapshot
curl -o backup.snapshot http://localhost:6333/collections/second_brain/snapshots/<snapshot_name>
```

### FalkorDB

FalkorDB data persists in the Docker volume. Back up the volume:

```bash
docker compose exec redis redis-cli BGSAVE
docker cp $(docker compose ps -q redis):/data ./falkordb-backup/
```
