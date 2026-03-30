# LLM Enrichment Pipeline

How RASPUTIN automatically improves and maintains memory quality using local LLMs at zero marginal cost.

## Overview

Four enrichment processes run continuously:

1. **A-MAC Quality Gate** — scores every commit before storage
2. **Entity Extraction** — extracts named entities on every commit, writes to FalkorDB
3. **Fact Extractor** — mines conversation sessions every 4h for personal facts
4. **Second Brain Enrichment** — 6x nightly, adds importance scores + tags to existing memories

---

## A-MAC Quality Gate

**A-MAC** = Adaptive Memory Admission Control. Prevents trivial, redundant, or low-value content from entering the vector store.

### How it works

Every `POST /commit` passes through `amac_gate()` before any storage happens:

```python
AMAC_THRESHOLD = 4.0
# model from config/rasputin.toml [amac] section
AMAC_TIMEOUT = 30                    # fail-open on timeout
```

The model is prompted to score the text on three dimensions:

```
Score this memory on 3 dimensions (0-10 each):
- Relevance: Is this genuinely useful information to remember?
- Novelty: Is this new information, not already obvious?
- Specificity: Does it contain concrete details (names, numbers, decisions)?

Reply with scores in format: R,N,S
```

**Composite = mean(Relevance, Novelty, Specificity)**

- Composite ≥ 4.0 → **ACCEPTED**, stored in Qdrant
- Composite < 4.0 → **REJECTED**, logged to `/tmp/amac_rejected.log`
- Timeout (30s) → **FAIL-OPEN**, accepted anyway (LLM busy = don't lose data)
- Diagnostic messages → **SKIPPED** (health check texts auto-bypass)

### Score parsing

The parser looks for the **last** valid `X,Y,Z` triplet in the model output. This works reliably across Qwen's thinking-style outputs where the model reasons before outputting the final scores.

```python
# Robust triplet parsing — finds LAST valid triplet
all_triplets = []
for line in lines:
    scores = re.findall(r'(?<!\d)(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})(?!\d)', line)
    for s in scores:
        if all(0 <= int(x) <= 10 for x in s):
            all_triplets.append(s)

# Use the last one (actual decision, not examples from reasoning)
r, n, s = all_triplets[-1]
```

### Metrics

The A-MAC module tracks running metrics accessible via `/stats`:

```json
{
  "amac_stats": {
    "accepted": 12450,
    "rejected": 3201,
    "bypassed": 44,
    "timeout_accepts": 12,
    "avg_score": 6.84
  }
}
```

A ~20% rejection rate is healthy — it means the gate is catching low-value content.

### Rejection log

Rejected memories are logged with full context for debugging:

```bash
tail -f /tmp/amac_rejected.log
# {"ts": "2026-03-15T14:22:01", "source": "conversation", 
#  "scores": {"relevance": 2.0, "novelty": 1.0, "specificity": 3.0, "composite": 2.0},
#  "text": "ok thanks"}
```

### Bypassing A-MAC

For bulk imports or when you know quality is high:

```bash
curl -X POST http://localhost:7777/commit \
  -d '{"text": "...", "source": "email", "force": true}'
```

---

## Entity Extraction

Every commit runs fast NER (Named Entity Recognition) to extract entities and write them to the FalkorDB knowledge graph.

### Fast NER (extract_entities_fast)

Regex-based pattern matching — no model required, runs in <1ms:

```python
def extract_entities_fast(text):
    entities = []
    
    # People: capitalized words not in skip list
    people = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
    
    # Money amounts
    money = re.findall(r'\$[\d,]+(?:\.\d+)?[KMB]?\b', text)
    
    # Dates
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{4}\b', text)
    
    # Organizations (all-caps or specific patterns)
    orgs = re.findall(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text)
    
    return list(set(people + money + dates + orgs))
```

### FalkorDB write (write_to_graph)

```python
def write_to_graph(point_id, text, entities, timestamp):
    # MERGE creates node if not exists, otherwise matches existing
    for entity_name in entities:
        g.query("""
            MERGE (e:Entity {name: $name})
            SET e.last_seen = $timestamp
            WITH e
            MERGE (m:Memory {id: $memory_id})
            SET m.timestamp = $timestamp, m.text_preview = $preview
            MERGE (e)-[:MENTIONED_IN]->(m)
        """, {
            "name": entity_name,
            "memory_id": str(point_id),
            "timestamp": timestamp,
            "preview": text[:100]
        })
```

---

## Fact Extractor

`tools/fact_extractor.py` runs every 4 hours via cron. It mines conversation session transcripts to extract structured personal facts about the user.

### Three-pass pipeline

**Pass 1: Extract** — LLM identifies candidate facts from raw conversation text
**Pass 2: Verify** — Second LLM pass validates each fact is genuine (not hypothetical)
**Pass 3: Filter** — Deduplication against existing facts.jsonl

### Setup

```bash
# Add to crontab
0 */4 * * * python3 tools/fact_extractor.py >> /tmp/fact_extractor.log 2>&1
```

### Configuration

Use `config/rasputin.toml` for default model/endpoint settings and env vars for overrides.

- `LLM_API_URL` (or config endpoint)
- `LLM_MODEL` (or config model)
- `SESSIONS_DIR`, `WORKSPACE_PATH`, and `QDRANT_URL` as needed per environment

### Running manually

```bash
# Process last 4 hours (normal cron mode)
python3 tools/fact_extractor.py

# Process last 24 hours
python3 tools/fact_extractor.py --hours 24

# First run: process ALL sessions
python3 tools/fact_extractor.py --all
```

### Output format (facts.jsonl)

```json
{"fact": "Team decided to move launch to April after dependency review", "confidence": 0.95, "source": "session_2026-03-15", "date": "2026-03-15T10:22:00", "hash": "abc123"}
{"fact": "Client requested weekly status updates and approved revised timeline", "confidence": 0.98, "source": "session_2026-02-20", "date": "2026-02-20T09:10:00", "hash": "def456"}
```

Facts are also committed to Qdrant with `source: "fact"` for retrieval in the main search pipeline.

### State tracking

The extractor maintains state to avoid reprocessing:

```json
{
  "last_run": "2026-03-15T10:00:00",
  "processed_lines": {"session_abc.jsonl": 1234},
  "fact_hashes": ["abc123", "def456"]
}
```

Hash deduplication prevents the same fact from being stored twice across runs.

---

## Second Brain Enrichment

`tools/enrich_second_brain.py` runs 6 times overnight (configurable via cron). For each memory:

1. Generates a quality importance score (0–100)
2. Extracts or refines tags
3. Identifies the memory category
4. Updates the Qdrant payload

```bash
# Enrich a batch of 100 memories
python3 tools/enrich_second_brain.py --batch 100

# Cron: 6x nightly at 1am, 2am, 3am, 4am, 5am, 6am
0 1-6 * * * python3 tools/enrich_second_brain.py --batch 100
```

---

## Auto-Tagging Pipeline

During enrichment, the configured local model generates structured tags for each memory:

**Prompt:**
```
Classify this memory with tags from these categories:
- domain: [business, health, personal, technical, research]  
- people: [names mentioned]
- urgency: [high, medium, low, none]
- action_required: [true, false]

Memory: {text}

Output JSON only.
```

**Output:**
```json
{
  "domain": "business",
  "people": ["Acme Corp"],
  "urgency": "medium",
  "action_required": false
}
```

Tags are stored in the Qdrant payload and available for filtered search.
