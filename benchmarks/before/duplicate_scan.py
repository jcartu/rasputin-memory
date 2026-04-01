#!/usr/bin/env python3
"""RASPUTIN Duplicate Memory Scan — BEFORE changes"""

import json
import urllib.request
import random
import statistics
from datetime import datetime

QDRANT = "http://localhost:6333"
OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/duplicate_scan.jsonl"
STATS_OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/duplicate_stats.json"

TOTAL_POINTS = 61959
SAMPLE_SIZE = 2000
SIM_THRESHOLD = 0.90

def qdrant_post(path, body):
    url = f"{QDRANT}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

print(f"Starting duplicate scan at {datetime.now().isoformat()}")
print(f"Sampling {SAMPLE_SIZE} points, threshold={SIM_THRESHOLD}")

# Step 1: Scroll through and collect point IDs + vectors
print("Collecting sample point IDs...")
all_ids = []
offset = None
while len(all_ids) < SAMPLE_SIZE * 3:  # collect more than needed for random selection
    body = {"limit": 250, "with_payload": False, "with_vector": False}
    if offset:
        body["offset"] = offset
    resp = qdrant_post("/collections/memories_v2/points/scroll", body)
    points = resp["result"]["points"]
    if not points:
        break
    all_ids.extend(p["id"] for p in points)
    offset = resp["result"].get("next_page_offset")
    if not offset:
        break
    if len(all_ids) >= 10000:
        break

print(f"Collected {len(all_ids)} point IDs")
random.shuffle(all_ids)
sample_ids = all_ids[:SAMPLE_SIZE]

# Step 2: For each sampled point, get its vector and search for near-duplicates
print(f"Scanning {len(sample_ids)} points for near-duplicates...")
duplicate_clusters = {}
checked = set()
dup_count = 0
batch_size = 20

with open(OUTPUT, "w") as out_f:
    for i, pt_id in enumerate(sample_ids):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(sample_ids)}, dups found so far: {dup_count}")
        
        if pt_id in checked:
            continue
        
        # Get vector for this point
        try:
            resp = qdrant_post("/collections/memories_v2/points", {"ids": [pt_id], "with_vector": True, "with_payload": True})
            if not resp["result"]:
                continue
            point = resp["result"][0]
            vector = point.get("vector")
            if not vector:
                continue
            payload = point.get("payload", {})
            text = payload.get("text", payload.get("content", ""))[:150]
        except Exception:
            continue
        
        # Search for near-duplicates
        try:
            search_resp = qdrant_post("/collections/memories_v2/points/search", {
                "vector": vector,
                "limit": 6,
                "score_threshold": SIM_THRESHOLD,
                "with_payload": True,
            })
            neighbors = search_resp.get("result", [])
        except Exception:
            continue
        
        # Filter out self (score ~1.0 with itself)
        true_dups = [n for n in neighbors if n["id"] != pt_id and n["score"] >= SIM_THRESHOLD]
        
        if true_dups:
            dup_count += len(true_dups)
            cluster = {
                "anchor_id": pt_id,
                "anchor_text": text,
                "duplicates": [
                    {
                        "id": n["id"],
                        "score": round(n["score"], 4),
                        "text": n["payload"].get("text", n["payload"].get("content", ""))[:150],
                    }
                    for n in true_dups
                ],
            }
            out_f.write(json.dumps(cluster) + "\n")
            duplicate_clusters[str(pt_id)] = cluster
            checked.update(n["id"] for n in true_dups)

print("\nDuplicate scan complete:")
print(f"  Points scanned: {len(sample_ids)}")
print(f"  Duplicate pairs/clusters found: {dup_count}")
print(f"  Anchor points with dups: {len(duplicate_clusters)}")

# Estimate total duplicates
dup_rate = dup_count / len(sample_ids) if sample_ids else 0
estimated_total_dups = int(dup_rate * TOTAL_POINTS)

# Score distribution among dups
if duplicate_clusters:
    all_scores = [d["score"] for c in duplicate_clusters.values() for d in c["duplicates"]]
    score_stats = {
        "mean": round(statistics.mean(all_scores), 4),
        "median": round(statistics.median(all_scores), 4),
        "min": round(min(all_scores), 4),
        "max": round(max(all_scores), 4),
        "count_090_095": sum(1 for s in all_scores if 0.90 <= s < 0.95),
        "count_095_099": sum(1 for s in all_scores if 0.95 <= s < 0.99),
        "count_099_100": sum(1 for s in all_scores if s >= 0.99),
    }
else:
    score_stats = {}

stats = {
    "benchmark_date": datetime.now().isoformat(),
    "sample_size": len(sample_ids),
    "threshold": SIM_THRESHOLD,
    "duplicate_pairs_in_sample": dup_count,
    "anchor_points_with_dups": len(duplicate_clusters),
    "duplicate_rate": round(dup_rate, 4),
    "estimated_total_duplicates_in_collection": estimated_total_dups,
    "total_collection_size": TOTAL_POINTS,
    "score_distribution": score_stats,
}

with open(STATS_OUTPUT, "w") as f:
    json.dump(stats, f, indent=2)

print(f"  Duplicate rate: {dup_rate:.2%}")
print(f"  Estimated total dups in 61K collection: {estimated_total_dups}")
print(f"\nSaved clusters to {OUTPUT}")
print(f"Saved stats to {STATS_OUTPUT}")
