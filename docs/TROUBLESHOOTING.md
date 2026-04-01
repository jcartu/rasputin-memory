# Troubleshooting Guide

Common issues, fixes, and performance tuning for the RASPUTIN memory system.

---

## 🔴 Critical: Embedding Model Mismatch

**Symptom:** Searches return nothing (or cosine ~0.63) even though you know matching memories exist.

**Cause:** You embedded new content with a different model than the stored vectors. If your collection was built with one embedding model and you query with another, vectors become geometrically incomparable.

**Fix:**
```bash
# Always use port 11434 for embeddings
curl -s http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": ["test"]}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['embeddings'][0]))"
# Should print: 768

# If you accidentally used the wrong model, delete the bad vectors
# They'll have a cosine similarity of ~0.63 with everything — identifiable
```

**Prevention:** Keep embedding configuration stable in `hybrid_brain.py`/`config/rasputin.toml`. Never change `EMBED_URL` or `EMBED_MODEL` once the collection is populated.

---

## 🔴 A-MAC Rejecting Everything

**Symptom:** All commits return `"status": "rejected"` with low scores.

**Possible causes:**

1. **Configured LLM endpoint is overloaded**
   ```bash
   # Check if the configured endpoint/model are responding
   # (read [amac].url and [amac].model from config/rasputin.toml)
   curl -s http://localhost:11434/v1/chat/completions \
     -d '{"model": "qwen2.5:14b", "messages": [{"role":"user","content":"hi"}], "max_tokens": 10}'
   ```
   If it's timing out (30s), A-MAC should fail-open. Check logs.

2. **Model not available** at the configured endpoint
   ```bash
   # Check model listing if your provider supports it
   curl -s http://localhost:11434/v1/models
   ```

3. **Threshold too high** — tune `[amac].threshold` in `config/rasputin.toml`

4. **Bypass for testing:**
   ```bash
   curl -X POST http://localhost:7777/commit \
     -d '{"text": "test memory", "force": true}'
   ```

---

## 🟡 Qdrant Not Starting

**Symptom:** `curl http://localhost:6333/collections` fails.

```bash
# Check if container is running
docker ps | grep qdrant

# Check logs
docker logs qdrant --tail 50

# Restart
docker restart qdrant

# If storage is corrupted, check the data dir
ls -la ~/.qdrant_storage/
```

**Storage permission issues:**
```bash
# Qdrant runs as non-root in Docker
sudo chown -R 1000:1000 ~/.qdrant_storage/
docker restart qdrant
```

---

## 🟡 FalkorDB Connection Refused

**Symptom:** Graph queries fail, `/health` shows `"falkordb": "error"`.

```bash
# Check container
docker ps | grep falkordb

# Test connection
redis-cli -p 6380 PING

# Check logs
docker logs falkordb --tail 50

# Common issue: wrong port mapping
# FalkorDB internal port is 6379, mapped to 6380
docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest
```

---

## 🟡 Reranker Unavailable

**Symptom:** Search works but results quality is poor. `/health` shows `"reranker": "unavailable"`.

The system falls back to vector-only ranking when the reranker is down — this is intentional. Searches will still work, just less precisely ranked.

```bash
# Check reranker
curl -s http://localhost:8006/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query": "test", "passages": ["hello"]}'

# If not running, restart it
python3 tools/reranker_server.py &

```

**GPU memory issues:**
```bash
# Check GPU memory
nvidia-smi

# The bge-reranker-v2-m3 needs ~2.3GB VRAM
# If OOM, use the smaller base model:
MODEL_NAME = "BAAI/bge-reranker-base"  # 0.5GB
```

---

## 🟡 Ollama Not Responding

**Symptom:** All searches fail with embedding errors.

```bash
# Check Ollama
systemctl status ollama
# or
ps aux | grep ollama

# Restart
systemctl restart ollama
# or
pkill ollama && ollama serve &

# Check the model is pulled
ollama list | grep nomic-embed-text

# Pull if missing
ollama pull nomic-embed-text
```

---

## 🟡 Searches Return Nothing

**Debugging checklist:**
```bash
# 1. Check Qdrant has data
curl http://localhost:6333/collections/second_brain | python3 -m json.tool
# Look for "vectors_count" > 0

# 2. Test embedding directly
curl -s http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": ["test"]}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print('OK, dim:', len(d['embeddings'][0]))"

# 3. Test raw Qdrant search
curl -X POST http://localhost:6333/collections/second_brain/points/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": ['"$(curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":["test"]}' | python3 -c "import json,sys; print(','.join(str(x) for x in json.load(sys.stdin)['embeddings'][0]))")"'], "limit": 3}'

# 4. Lower the threshold temporarily
curl "http://localhost:7777/search?q=test&limit=5&threshold=0.3"
```

---

## 🟡 Fact Extractor Not Running

**Symptom:** `memory/facts.jsonl` not being updated.

```bash
# Check cron
crontab -l | grep fact_extractor

# Run manually and check output
python3 tools/fact_extractor.py --hours 4

# Check state file
cat memory/fact_extractor_state.json

# Check sessions directory exists and has files
ls ./sessions/ | head -5

# If state is corrupted, reset it
echo '{"last_run": null, "processed_lines": {}, "fact_hashes": []}' > memory/fact_extractor_state.json
```

---

## Performance Tuning

### Slow searches (>500ms)

The bottleneck is almost always the embedding call. Benchmark:

```bash
time curl -s http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": ["test query about LATAM"]}' > /dev/null
```

If >100ms, Ollama may be CPU-only. Check GPU usage:
```bash
nvidia-smi dmon -s u -d 1
# Watch for "nomic-embed" VRAM allocation
```

**Ollama GPU acceleration:**
```bash
# Check if Ollama is using GPU
OLLAMA_GPU_OVERHEAD=0 ollama run nomic-embed-text "test" 2>&1 | grep -i gpu
```

### Reduce reranker latency

The reranker processes all candidates in a single batch. Reduce the candidate pool:

```python
RERANKER_CANDIDATE_POOL = 20  # Default 50 — halving this saves ~50ms
```

### Qdrant index optimization

After bulk imports, force Qdrant to optimize its index:

```bash
curl -X POST http://localhost:6333/collections/second_brain/index
```

### BM25 cold start

BM25 builds an in-memory index from Qdrant data. On first search after restart, it may take a few seconds. Pre-warm it:

```bash
curl "http://localhost:7777/search?q=warmup&limit=1"
```

---

## Checking System Health

```bash
# Full health check
curl -s http://localhost:7777/health | python3 -m json.tool

# Stats dashboard
curl -s http://localhost:7777/stats | python3 -m json.tool

# Search API recall test
curl -s "http://localhost:7777/search?q=test+query&limit=3" | python3 -m json.tool

# Check all Docker containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

```

---

## Log Locations

| Log | Location | Contents |
|-----|---------|---------|
| hybrid_brain | service stdout/stderr logs | API requests, A-MAC decisions |
| A-MAC rejections | `/tmp/amac_rejected.log` | Rejected commits with scores |
| Fact extractor | `/tmp/fact_extractor.log` | Cron runs, extracted facts |
| Enrichment | `/tmp/enrichment.log` | Nightly enrichment runs |
| Qdrant | `docker logs qdrant` | Database operations |
| FalkorDB | `docker logs falkordb` | Graph operations |
| Ollama | `journalctl -u ollama` | Embedding service |
