# Benchmark Results — 2026-04-02

| Category | Recall@5 | Recall@10 | MRR@10 | Latency p50 |
|----------|----------|-----------|--------|-------------|
| overall | 0.67 | 0.74 | 0.56 | 28ms |
| contradiction | 0.48 | 0.60 | 0.48 | 31ms |
| decay | 0.23 | 0.40 | 0.16 | 21ms |
| dedup | 1.00 | 1.00 | 1.00 | 25ms |
| edge_cases | 0.67 | 0.67 | 0.63 | 28ms |
| entity | 0.20 | 0.43 | 0.11 | 28ms |
| multilingual | 0.97 | 0.97 | 0.55 | 30ms |
| recency | 1.00 | 1.00 | 0.80 | 33ms |
| source_tier | 1.00 | 1.00 | 1.00 | 30ms |

- Supersede rate (contradiction): 0.92
- Dedup precision: 1.00
