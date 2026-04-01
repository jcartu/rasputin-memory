# RASPUTIN Memory — Before vs After Benchmark Comparison

**Before:** 2026-03-31T15:51:53.534153
**After:** 2026-03-31T16:48:22.864007
**System:** hybrid_brain_v5 (Phase 4)

## 1. Search Quality (50 queries)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| MRR | 0.9183 | 0.2700 | -0.6483 ⬇️ |
| P@1 | 0.9000 | 0.2200 | -0.6800 ⬇️ |
| P@3 | 0.8467 | 0.1867 | -0.6600 ⬇️ |
| P@5 | 0.8360 | 0.1400 | -0.6960 ⬇️ |
| P@10 | 0.8140 | 0.0740 | -0.7400 ⬇️ |

### Latency (Search)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| median latency ms | 180.8ms | 75.6ms | -105.3ms ✅ |
| p95 latency ms | 204.6ms | 94.1ms | -110.5ms ✅ |
| p99 latency ms | 214.3ms | 121.5ms | -92.8ms ✅ |
| min latency ms | 150.3ms | 61.3ms | -89.0ms ✅ |
| max latency ms | 214.3ms | 121.5ms | -92.8ms ✅ |

### Per-Category MRR

| Category | Before MRR | After MRR | Delta | Before P@1 | After P@1 |
|----------|-----------|----------|-------|-----------|----------|
| personal | 1.000 | 0.667 | -0.333 ⬇️ | 1.000 | 0.500 |
| business | 1.000 | 0.150 | -0.850 ⬇️ | 1.000 | 0.100 |
| technical | 1.000 | 0.000 | -1.000 ⬇️ | 1.000 | 0.000 |
| people | 0.833 | 0.333 | -0.500 ⬇️ | 0.800 | 0.300 |
| temporal | 0.850 | 0.400 | -0.450 ⬇️ | 0.800 | 0.400 |
| edge_cases | 0.667 | 0.000 | -0.667 ⬇️ | 0.600 | 0.000 |

## 2. Endpoint Performance

| Endpoint | Before Mean | After Mean | Delta | Before P95 | After P95 | Delta P95 |
|----------|------------|-----------|-------|-----------|----------|---------|
| `GET /health` | 141.5ms | 40.7ms | -100.8ms ✅ | 146.7ms | 44.8ms | -101.9ms |
| `GET /search?q=test&limit=5` | 159.4ms | 45.9ms | -113.6ms ✅ | 183.9ms | 70.6ms | -113.3ms |
| `GET /search?q=Josh+wife&limit=10` | 168.9ms | 2.5ms | -166.4ms ✅ | 189.0ms | 2.1ms | -186.8ms |
| `GET /stats` | 7.7ms | 6.1ms | -1.6ms ➡️ | 11.5ms | 7.6ms | -3.9ms |
| `POST /commit` | 3288.7ms | 350.4ms | -2938.2ms ✅ | — | — | — |

## 3. Health Stats

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Qdrant total points | 61,959 | 61,959 | +0 |
| FalkorDB nodes | 107,320 | 107,320 | +0 |
| FalkorDB edges | 124,792 | 124,792 | +0 |

## 4. Summary

- **MRR change:** -0.6483 (-70.6%)
- **P@1 change:** -0.6800
- **Median latency change:** -105.3ms
- **P95 latency change:** -110.5ms
- **Qdrant points delta:** +0