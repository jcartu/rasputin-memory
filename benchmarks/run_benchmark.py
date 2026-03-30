#!/usr/bin/env python3
"""Benchmark runner for recall quality and latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * p))))
    return sorted_values[idx]


def load_ground_truth(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def server_available(api_url: str, timeout: float = 2.0) -> bool:
    try:
        resp = requests.get(f"{api_url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def run_benchmarks(ground_truth_path: str, api_url: str = "http://localhost:7777") -> int:
    dataset_path = Path(ground_truth_path)
    if not dataset_path.exists():
        print(f"[SKIP] Ground-truth file not found: {dataset_path}")
        return 0

    if not server_available(api_url):
        print(f"[SKIP] Server unavailable at {api_url}; skipping benchmark run.")
        return 0

    queries = load_ground_truth(dataset_path)
    if not queries:
        print("[SKIP] Ground-truth dataset is empty.")
        return 0

    latencies_ms: list[float] = []
    recall_hits = 0
    mrr_sum = 0.0
    ok_count = 0

    for q in queries:
        query = q.get("query", "")
        expected_keywords = [str(x).lower() for x in q.get("expected_keywords", [])]
        if not query or not expected_keywords:
            continue

        start = time.perf_counter()
        try:
            resp = requests.get(f"{api_url}/search", params={"q": query, "limit": 5}, timeout=20)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
            if resp.status_code != 200:
                continue
            payload = resp.json()
        except Exception:
            continue

        ok_count += 1
        results = payload.get("results", [])
        rank = None
        for i, item in enumerate(results):
            text = str(item.get("text", "")).lower()
            if any(keyword in text for keyword in expected_keywords):
                rank = i + 1
                break

        if rank is not None:
            recall_hits += 1
            mrr_sum += 1.0 / rank

    if ok_count == 0:
        print("[SKIP] No successful responses from /search.")
        return 0

    latencies_ms.sort()
    recall_at_5 = recall_hits / ok_count
    mrr = mrr_sum / ok_count

    print("=== RASPUTIN Benchmark ===")
    print(f"Dataset: {dataset_path}")
    print(f"API: {api_url}")
    print(f"Queries attempted: {len(queries)}")
    print(f"Successful responses: {ok_count}")
    print(f"Recall@5: {recall_hits}/{ok_count} ({recall_at_5 * 100:.1f}%)")
    print(f"MRR: {mrr:.3f}")
    print(f"Latency mean: {statistics.mean(latencies_ms):.1f}ms")
    print(f"Latency p50: {percentile(latencies_ms, 0.50):.1f}ms")
    print(f"Latency p95: {percentile(latencies_ms, 0.95):.1f}ms")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval quality and latency benchmarks")
    parser.add_argument("--ground-truth", default="benchmarks/ground_truth.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--api-url", default="http://localhost:7777", help="Hybrid API URL")
    args = parser.parse_args()
    return run_benchmarks(ground_truth_path=args.ground_truth, api_url=args.api_url)


if __name__ == "__main__":
    raise SystemExit(main())
