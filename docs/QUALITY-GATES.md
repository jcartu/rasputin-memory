# Quality Gates

## A-MAC Scoring

Every memory is scored by a local LLM before storage:

| Dimension | What It Measures |
|-----------|-----------------|
| **Relevance** | Is this useful for future retrieval? |
| **Novelty** | Does this add new information vs. what's already stored? |
| **Specificity** | Is this concrete and actionable, not vague? |

**Composite score < 4.0 → rejected.** This prevents the memory system from filling up with low-value noise.

The scoring is performed by `tools/enrich_second_brain.py` as part of the enrichment pipeline. See [Enrichment Pipeline](ENRICHMENT.md) for the full details on A-MAC scoring, entity extraction, and auto-tagging.

## Ebbinghaus Decay

Memory strength degrades over time following an Ebbinghaus-inspired forgetting curve:

- **Recently accessed memories** stay strong — access resets the decay timer
- **Consolidated memories** (extracted facts) resist decay — they've proven their value
- **Unconsolidated, unaccessed memories** gradually fade — preventing stale context from polluting search results

The decay system is implemented in `tools/memory_decay.py` and runs on a scheduled basis.

## How They Work Together

```
New Memory → A-MAC Score ≥ 4.0? → Store → Decay over time → Consolidate survivors
                    ↓ No
               Rejected (not stored)
```

Quality gates ensure that only high-value memories enter the system, and temporal decay ensures that only actively useful memories remain prominent in search results.
