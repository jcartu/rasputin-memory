# Failure Taxonomy Report

**Total:** 199 questions, 102 correct (51.3%)
**Gold-in-ANY-chunk:** 176/199 (88.4%)
**Gold-in-Top-5:** 127/199 (63.8%)
**Gold-in-Top-10:** 142/199 (71.4%)

## Retrieval Metrics by Category

| Category | Total | Accuracy | Gold-ANY | Gold-Top5 | Gold-Top10 | MRR |
|----------|-------|----------|----------|-----------|------------|-----|
| adversarial | 47 | 6.4% | 96% | 72% | 79% | 0.665 |
| multi-hop | 13 | 61.5% | 46% | 15% | 23% | 0.129 |
| open-domain | 70 | 77.1% | 96% | 67% | 76% | 0.576 |
| single-hop | 32 | 46.9% | 78% | 44% | 56% | 0.355 |
| temporal | 37 | 59.5% | 89% | 81% | 84% | 0.679 |

## Failure Taxonomy (incorrect answers only)

| Category | Wrong | Retrieval Miss | Retrieval Buried | Generation Fail |
|----------|-------|----------------|------------------|-----------------|
| adversarial | 44 | 2 (4%) | 8 (18%) | 34 (77%) |
| multi-hop | 5 | 4 (80%) | 1 (20%) | 0 (0%) |
| open-domain | 16 | 3 (18%) | 6 (37%) | 7 (43%) |
| single-hop | 17 | 6 (35%) | 3 (17%) | 8 (47%) |
| temporal | 15 | 3 (20%) | 1 (6%) | 11 (73%) |