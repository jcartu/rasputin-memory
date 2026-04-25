# Embedding A/B: qwen3-embedding-8b vs nomic-embed-text

**Date:** 2026-04-05
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense-only (all pipeline stages disabled)

## Hypothesis

qwen3-embedding-8b (#1 MTEB multilingual, 4096d truncated to 768d via Matryoshka)
will improve retrieval quality over nomic-embed-text (768d native).

## Setup

- **Baseline:** nomic-embed-text on local Ollama (22ms/embed)
- **Test:** qwen3-embedding:8b on a remote inference host (~200ms/embed over LAN)
- Both use Qdrant collections with 768d vectors
- qwen3 output truncated from 4096d to 768d + L2 normalized

## Results

| Model | Accuracy | Gold-ANY | Gold-Top5 | Gold-Top10 |
|-------|----------|----------|-----------|------------|
| nomic-embed-text (768d) | 65.1% | 88.4% | 63.8% | 71.4% |
| qwen3-embedding-8b (4096→768d) | 65.1% | 88.4% | 63.8% | 71.4% |

Per-category: identical across all 5 categories.
Failure taxonomy: identical.

## Verdict: NO DIFFERENCE

qwen3-embedding truncated to 768d is equivalent to nomic-embed-text at 768d.

Possible explanations:
1. Matryoshka truncation from 4096→768 loses the quality advantage
2. For conversational memory (short, informal text), both models embed equally well
3. The retrieval ceiling is set by chunking strategy and query formulation, not embedding quality
4. Need to test at full 4096d (requires recreating Qdrant collections)

## Next Steps

Test qwen3-embedding at native 4096d — requires changing collection dimensions.
The chunking strategy (5-turn windows, stride 2) may be the real bottleneck.
