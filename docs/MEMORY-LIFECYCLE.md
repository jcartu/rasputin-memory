# Memory Lifecycle

```
Create → Score → Store → Search → Decay → Consolidate → Deduplicate
```

| Stage | What Happens |
|-------|-------------|
| **Create** | New memory enters the pipeline (conversation, fact, decision) |
| **Score** | A-MAC quality gate evaluates Relevance, Novelty, Specificity (composite ≥ 4.0 to pass) |
| **Store** | Embedded into Qdrant + entities extracted into FalkorDB knowledge graph |
| **Search** | 4-stage hybrid retrieval on every query |
| **Decay** | Ebbinghaus-inspired strength decay — older, unconsolidated memories fade over time |
| **Consolidate** | Weekly fact extraction from daily logs — 7K+ facts per run via direct GPU routing |
| **Deduplicate** | Cosine similarity threshold (0.92) — identified and removed 24K+ duplicate vectors |

## Create

New memories enter the pipeline from multiple sources:
- Direct conversation commits via the `/commit` API endpoint
- Automated extraction from daily logs
- Fact extraction from session transcripts (runs every 4h via cron)

## Score (A-MAC Quality Gate)

Every memory is scored by a local LLM before storage across three dimensions:

| Dimension | What It Measures |
|-----------|-----------------|
| **Relevance** | Is this useful for future retrieval? |
| **Novelty** | Does this add new information vs. what's already stored? |
| **Specificity** | Is this concrete and actionable, not vague? |

**Composite score < 4.0 → rejected.** This prevents the memory system from filling up with low-value noise.

See also: [Enrichment Pipeline](ENRICHMENT.md) for full A-MAC scoring details.

## Store

Accepted memories are:
1. Embedded into a 768-dimensional vector via `nomic-embed-text`
2. Stored in Qdrant with metadata (source, timestamp, tags)
3. Entities extracted and added to the FalkorDB knowledge graph

## Search

4-stage hybrid retrieval runs on every query — see [Architecture](ARCHITECTURE.md).

## Decay (Ebbinghaus Forgetting Curve)

Memory strength degrades over time following an Ebbinghaus-inspired forgetting curve:
- Recently accessed memories stay strong
- Consolidated memories (extracted facts) resist decay
- Unconsolidated, unaccessed memories gradually fade
- Prevents stale context from polluting search results

## Consolidate

Weekly fact extraction from daily logs:
- 5-pass extraction via `memory_consolidate.py`
- Parallel variant via `memory_consolidator_v4.py` for session transcripts
- Produces 7K+ structured facts per run

## Deduplicate

Cosine similarity dedup (threshold: 0.92):
- Scans for near-duplicate vectors
- Identified and removed 24K+ duplicates in production
- Run via `memory_dedup.py`
