# Step 1: Cohere Reranker Verification

**Date:** 2026-04-05
**Dataset:** LoCoMo conv-0, 199 questions

## Hypothesis

The Cohere reranker may not have been active during the dense-only ablation.
Need to verify it was running and confirm whether it adds 0pp.

## Findings

1. **Key is present** in `.env` (COHERE_API_KEY=LlkJM...) but NOT auto-loaded by Python.
   It must be passed explicitly as an env var.

2. **Explicit Cohere test:** Ran with RERANK_PROVIDER=cohere, COHERE_API_KEY set,
   all other pipeline stages OFF. Search calls took ~2.8s each (Cohere API round-trip).

3. **Results identical** to no-Cohere run:
   - Every per-category accuracy score matched exactly
   - Every failure taxonomy metric matched exactly
   - The reranker is called but does not change the rank order enough to affect
     which chunks the answer model receives

4. **Why Cohere adds nothing:** The benchmark uses top-60 chunks with a weak answer
   model. Cohere may reorder positions 1-60 but the answer model sees all 60 anyway.
   The reranker would matter more with top-5 or top-10 context windows.

## Verdict

Cohere reranker is confirmed active when key is provided, but adds 0pp for this
benchmark configuration (60-chunk context). Reranker would need to be tested with
smaller context windows to show value.
