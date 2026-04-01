# LoCoMo Benchmark Results — RASPUTIN Memory

**Date:** 2026-04-02 02:28 UTC
**Total QA pairs:** 1986

## All Runs Comparison

| Config | Overall F1 | single-hop | multi-hop | temporal | open-domain | adversarial |
|--------|-----------|------------|-----------|----------|-------------|-------------|
| nomic-embed 768d (v1) | 0.4144 | 0.3956 | 0.4651 | 0.2031 | 0.5841 | 0.1153 |
| nomic-embed 768d (v2, improved prompt) | 0.4417 | 0.4003 | 0.4479 | 0.2089 | 0.6042 | 0.2073 |
| qwen3-embed 4096d | ~0.18 | — | — | — | — | — |
| qwen3-embed 1024d (Matryoshka) | ~0.26 | — | — | — | — | — |

## Leaderboard

| System | F1 Score |
|--------|----------|
| memmachine | 0.8487 |
| zep | 0.7514 |
| mem0 | 0.6688 |
| rasputin-nomic-v2 | 0.4417 ⬅️ |
| rasputin-nomic-v1 | 0.4144 |
| rasputin-qwen3embed-1024d | 0.2564 |
| rasputin-qwen3embed-4096d | 0.1824 |

## Key Findings

### Qwen3-Embedding is BAD for asymmetric retrieval
- nomic-embed-text (768d): relevant/irrelevant cosine diff = 0.22
- qwen3-embedding (4096d): relevant/irrelevant cosine diff = -0.006 (ZERO discrimination)
- qwen3-embedding (1024d Matryoshka): diff = 0.03 (slightly better but still awful)
- Qwen3-Embedding appears optimized for symmetric similarity, not query→passage retrieval

### Prompt improvements
- Removed 'say unknown' instruction → adversarial should improve
- Added temporal sorting + explicit date prefix → temporal should improve
- Changed to 'make your best inference' prompt
