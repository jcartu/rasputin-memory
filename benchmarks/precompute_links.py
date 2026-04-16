#!/usr/bin/env python3
"""Pre-compute semantic kNN links for all points in a Qdrant collection.

For each point with a vector, finds top-K nearest neighbors (cosine >= threshold)
and stores their IDs in the point's payload as ``similar_ids``.

Usage:
    python3 benchmarks/precompute_links.py --collection locomo_lb_conv_26
    python3 benchmarks/precompute_links.py --collection locomo_lb_conv_26 --top-k 30 --threshold 0.5
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def http_json(url, data=None, method=None, timeout=60):
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def scroll_all_points(collection: str) -> list[dict]:
    all_points: list[dict] = []
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        data = http_json(
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            data=body,
            method="POST",
            timeout=30,
        )
        points = data.get("result", {}).get("points", [])
        for p in points:
            if p.get("vector"):
                all_points.append(
                    {
                        "id": p["id"],
                        "vector": p["vector"],
                        "chunk_type": p.get("payload", {}).get("chunk_type", ""),
                    }
                )
        next_offset = data.get("result", {}).get("next_page_offset")
        if not next_offset:
            break
        offset = next_offset
    return all_points


def compute_links(collection: str, top_k: int, threshold: float) -> int:
    points = scroll_all_points(collection)
    if not points:
        print(f"  {collection}: no points with vectors, skipping")
        return 0

    print(f"  {collection}: computing kNN links for {len(points)} points (top_k={top_k}, threshold={threshold})...")
    total_links = 0
    t0 = time.monotonic()

    for i, point in enumerate(points):
        try:
            data = http_json(
                f"{QDRANT_URL}/collections/{collection}/points/query",
                data={
                    "query": point["vector"],
                    "limit": top_k + 1,
                    "with_payload": False,
                    "score_threshold": threshold,
                },
                method="POST",
                timeout=10,
            )

            similar_ids = []
            for p in data.get("result", {}).get("points", []):
                if p["id"] != point["id"] and p.get("score", 0) >= threshold:
                    similar_ids.append(p["id"])
            similar_ids = similar_ids[:top_k]

            if similar_ids:
                http_json(
                    f"{QDRANT_URL}/collections/{collection}/points/payload",
                    data={
                        "points": [point["id"]],
                        "payload": {"similar_ids": similar_ids},
                    },
                    method="POST",
                    timeout=10,
                )
                total_links += len(similar_ids)

        except Exception as e:
            if i < 3:
                print(f"    Link error for point {point['id']}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(points) - i - 1) / rate if rate > 0 else 0
            print(f"    {i + 1}/{len(points)} processed, {total_links} links, ETA {remaining:.0f}s")

    elapsed = time.monotonic() - t0
    print(f"  {collection}: {total_links} links created for {len(points)} points in {elapsed:.1f}s")
    return total_links


def main():
    parser = argparse.ArgumentParser(description="Precompute kNN links for a Qdrant collection")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--top-k", type=int, default=30, help="Neighbors per point (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum cosine similarity (default: 0.5)")
    args = parser.parse_args()

    total = compute_links(args.collection, args.top_k, args.threshold)
    print(f"Total links created: {total}")


if __name__ == "__main__":
    main()
