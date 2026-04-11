# Multi-hop Retrieval Miss Analysis

**Date:** 2026-04-11
**Dataset:** Phase 1 full 10-conv results (1986 questions)
**Focus:** 43 multi-hop failures

## Diagnosis

| Diagnosis | Count | % | Description |
|-----------|-------|---|-------------|
| EXTRACTION_MISS | 26 | 60% | Gold answer content not in ANY fact or window |
| EMBEDDING_MISS | 17 | 40% | Content exists but cosine < 0.12 to query |

## Key Findings

1. **No ranking failures.** Zero cases where gold is in top-60 but ranked wrong.
   The cross-encoder is doing its job — it ranks well what it receives.

2. **60% extraction misses.** Multi-hop gold answers require INFERENCE from
   scattered facts. "Would Melanie be considered LGBTQ?" has no single fact
   saying "Melanie is not LGBTQ" — it must be inferred from absence of evidence.
   Fact extraction captures explicit statements, not implicit conclusions.

3. **40% embedding misses.** Gold exists in Qdrant but cosine similarity to
   the query is extremely low (all < 0.12). Example: "What Console does Nate
   own?" → cosine 0.099 to fact mentioning "Xenoblade 2" (a Switch game).
   Dense search requires semantic overlap between query and document phrasing.

4. **All cosine scores to gold are < 0.13.** The similarity floor (0.3 in Qdrant
   query) filters out ALL gold facts for multi-hop. Even without the floor,
   these would rank below hundreds of other facts.

## Implications

- **Observations can't fix extraction misses.** If the raw fact doesn't exist,
  no consolidation can create it. Only the original conversation text has it.
- **Entity search can't fix embedding misses.** Entity match finds facts about
  the right person, but the specific fact still has low query relevance.
- **Graph expansion can't fix either.** Following links from top-5 seeds goes
  sideways (semantically similar), not toward the gold fact.

## What Would Fix This

1. **Compare mode** — allows the answer model to see the full conversation
   context, bypassing the retrieval bottleneck entirely.
2. **Better fact extraction** — extract inferential conclusions, not just
   explicit statements. "Based on this conversation, Melanie appears to be
   supportive of LGBTQ causes but does not identify as LGBTQ herself."
3. **Hybrid dense+sparse retrieval** — BM25 keyword match for specific terms
   like "Xenoblade" or "Nintendo Switch" that dense vectors miss.
