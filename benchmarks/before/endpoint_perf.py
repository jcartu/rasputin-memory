#!/usr/bin/env python3
"""RASPUTIN Endpoint Performance Benchmark — BEFORE changes"""

import json
import time
import urllib.request
import urllib.parse
import statistics
from datetime import datetime

BASE = "http://localhost:7777"
QDRANT = "http://localhost:6333"
OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/endpoint_performance.json"

def timed_get(url, n=100):
    times = []
    errors = 0
    for _ in range(n):
        start = time.time()
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                r.read()
        except Exception:
            errors += 1
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        time.sleep(0.02)
    return times, errors

def timed_post(url, body, n=10):
    times = []
    errors = 0
    data = json.dumps(body).encode()
    for _ in range(n):
        start = time.time()
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
        except Exception as e:
            errors += 1
            print(f"  POST error: {e}")
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        time.sleep(0.1)
    return times, errors

def stats_for(times):
    if not times:
        return {}
    s = sorted(times)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p50_ms": round(s[int(n * 0.50)], 2),
        "p95_ms": round(s[int(n * 0.95)], 2),
        "p99_ms": round(s[min(int(n * 0.99), n-1)], 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "stdev_ms": round(statistics.stdev(times) if n > 1 else 0, 2),
        "errors": 0,
    }

results = {}
print(f"Starting endpoint performance at {datetime.now().isoformat()}")

print("Testing GET /health (100x)...")
times, errors = timed_get(f"{BASE}/health", 100)
r = stats_for(times)
r["errors"] = errors
results["GET /health"] = r
print(f"  mean={r['mean_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

print("Testing GET /search?q=test&limit=5 (100x)...")
times, errors = timed_get(f"{BASE}/search?q=test&limit=5", 100)
r = stats_for(times)
r["errors"] = errors
results["GET /search?q=test&limit=5"] = r
print(f"  mean={r['mean_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

print("Testing GET /search?q=Josh+wife&limit=10 (100x)...")
times, errors = timed_get(f"{BASE}/search?q=Josh+wife&limit=10", 100)
r = stats_for(times)
r["errors"] = errors
results["GET /search?q=Josh+wife&limit=10"] = r
print(f"  mean={r['mean_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

print("Testing GET /stats (100x)...")
times, errors = timed_get(f"{BASE}/stats", 100)
r = stats_for(times)
r["errors"] = errors
results["GET /stats"] = r
print(f"  mean={r['mean_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

# POST /commit — 10x with test data, then clean up
print("Testing POST /commit (10x)...")
test_text_prefix = "BENCHMARK_TEST_MEMORY_DELETE_ME_"
commit_ids = []
commit_times = []

for i in range(10):
    test_body = {
        "text": f"{test_text_prefix}{i} This is a temporary benchmark test memory that should be deleted.",
        "source": "benchmark_test",
        "importance": 0,
    }
    start = time.time()
    try:
        data = json.dumps(test_body).encode()
        req = urllib.request.Request(f"{BASE}/commit", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_data = json.loads(resp.read())
            # Try to get the ID from response
            pt_id = resp_data.get("id", resp_data.get("point_id", resp_data.get("ids", [None])))
            if isinstance(pt_id, list):
                pt_id = pt_id[0] if pt_id else None
            if pt_id:
                commit_ids.append(pt_id)
    except Exception as e:
        print(f"  commit error: {e}")
    elapsed = (time.time() - start) * 1000
    commit_times.append(elapsed)
    time.sleep(0.1)

r = stats_for(commit_times)
r["committed_ids_for_cleanup"] = commit_ids
results["POST /commit"] = r
print(f"  mean={r['mean_ms']:.1f}ms  committed {len(commit_ids)} IDs")

# Cleanup: delete test points from Qdrant
if commit_ids:
    print(f"Cleaning up {len(commit_ids)} test points...")
    try:
        delete_body = {"points": commit_ids}
        data = json.dumps(delete_body).encode()
        req = urllib.request.Request(
            f"{QDRANT}/collections/memories_v2/points/delete",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            del_resp = json.loads(resp.read())
            print(f"  Qdrant delete response: {del_resp}")
    except Exception as e:
        print(f"  Cleanup error: {e} — may need manual cleanup")
        print(f"  IDs to delete: {commit_ids}")
else:
    print("No IDs returned from commit — searching for test memories to clean up...")
    # Search for them and delete by filter
    try:
        filter_body = {
            "filter": {
                "must": [{"key": "source", "match": {"value": "benchmark_test"}}]
            }
        }
        data = json.dumps(filter_body).encode()
        req = urllib.request.Request(
            f"{QDRANT}/collections/memories_v2/points/delete",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            del_resp = json.loads(resp.read())
            print(f"  Filter-delete response: {del_resp}")
    except Exception as e:
        print(f"  Filter cleanup error: {e}")

output = {
    "benchmark_date": datetime.now().isoformat(),
    "results": results,
}

with open(OUTPUT, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {OUTPUT}")
print("\nSUMMARY:")
for ep, s in results.items():
    if ep != "POST /commit":
        print(f"  {ep:<40} mean={s['mean_ms']:>7.1f}ms  p95={s['p95_ms']:>7.1f}ms  p99={s['p99_ms']:>7.1f}ms")
