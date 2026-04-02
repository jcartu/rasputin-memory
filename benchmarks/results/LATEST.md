# Benchmark Results — 2026-04-02

| Category | Recall@5 | Recall@10 | MRR@10 | Latency p50 |
|----------|----------|-----------|--------|-------------|
| overall | 0.92 | 0.96 | 0.75 | 29ms |
| contradiction | 1.00 | 1.00 | 0.72 | 30ms |
| decay | 1.00 | 1.00 | 0.83 | 22ms |
| dedup | 1.00 | 1.00 | 1.00 | 24ms |
| edge_cases | 0.67 | 0.67 | 0.67 | 33ms |
| entity | 0.63 | 0.97 | 0.32 | 28ms |
| multilingual | 0.97 | 0.97 | 0.60 | 30ms |
| recency | 1.00 | 1.00 | 1.00 | 29ms |
| source_tier | 1.00 | 1.00 | 1.00 | 31ms |

- Supersede rate (contradiction): 1.00
- Dedup precision: 1.00
