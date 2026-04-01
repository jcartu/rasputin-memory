# Benchmarks

## Latency Benchmarks

Run a quick API round-trip for end-to-end latency measurements:

```bash
time curl -s "http://localhost:7777/health" > /dev/null
time curl -s "http://localhost:7777/search?q=latency+benchmark&limit=5" > /dev/null
```

This tests commit → search round-trip latency across all pipeline stages.

## Quality Benchmarks

Quality benchmarks require a ground-truth dataset specific to your memory corpus. To run your own comparison:

1. Prepare a JSONL file with `{"query": "...", "expected_ids": [...]}` entries
2. Run queries against the `/search` endpoint
3. Measure Recall@K and MRR against your ground truth

No pre-built quality benchmarks are included — results depend heavily on your data distribution and use case.
