# Benchmark Results — 2026-04-02

| Category | Recall@5 | Recall@10 | MRR@10 | Latency p50 |
|----------|----------|-----------|--------|-------------|
| overall | 0.82 | 0.89 | 0.68 | 29ms |
| contradiction | 0.96 | 0.96 | 0.71 | 30ms |
| decay | 0.40 | 0.50 | 0.36 | 23ms |
| dedup | 1.00 | 1.00 | 1.00 | 23ms |
| edge_cases | 0.67 | 0.67 | 0.67 | 32ms |
| entity | 0.63 | 0.97 | 0.32 | 28ms |
| multilingual | 0.97 | 0.97 | 0.60 | 30ms |
| recency | 1.00 | 1.00 | 1.00 | 30ms |
| source_tier | 1.00 | 1.00 | 1.00 | 31ms |

- Supersede rate (contradiction): 1.00
- Dedup precision: 1.00
