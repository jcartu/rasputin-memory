#!/usr/bin/env python3
"""Pre-compute semantic kNN links for all facts in LoCoMo collections.

For each fact, finds top-5 nearest neighbors (cosine >= 0.7) and stores
links in the point's payload as 'semantic_links'. Phase 3 graph expansion
follows these links to surface connected facts during search.

Usage:
    python3 benchmarks/precompute_links.py [--conversations 0,1,2]
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCOMO_FILE = REPO / "benchmarks" / "locomo" / "locomo10.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

LINK_TOP_K = 5
LINK_THRESHOLD = 0.7


def http_json(url, data=None, method=None, timeout=60):
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def scroll_all_points(collection):
    all_points = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        data = http_json(f"{QDRANT_URL}/collections/{collection}/points/scroll", data=body, method="POST", timeout=30)
        points = data.get("result", {}).get("points", [])
        for p in points:
            payload = p.get("payload", {})
            if payload.get("chunk_type") == "fact" and p.get("vector"):
                all_points.append(
                    {
                        "id": p["id"],
                        "vector": p["vector"],
                        "text": payload.get("text", "")[:100],
                    }
                )
        next_offset = data.get("result", {}).get("next_page_offset")
        if not next_offset:
            break
        offset = next_offset
    return all_points


def create_links_for_collection(collection):
    points = scroll_all_points(collection)
    if not points:
        print(f"  {collection}: no facts with vectors, skipping")
        return 0

    print(f"  {collection}: computing links for {len(points)} facts...")
    total_links = 0

    for i, point in enumerate(points):
        try:
            data = http_json(
                f"{QDRANT_URL}/collections/{collection}/points/query",
                data={
                    "query": point["vector"],
                    "limit": LINK_TOP_K + 1,
                    "with_payload": False,
                    "score_threshold": LINK_THRESHOLD,
                },
                method="POST",
                timeout=10,
            )

            links = []
            for p in data.get("result", {}).get("points", []):
                if p["id"] != point["id"] and p.get("score", 0) >= LINK_THRESHOLD:
                    links.append({"to_id": str(p["id"]), "sim": round(p["score"], 4)})

            if links:
                links = links[:LINK_TOP_K]
                http_json(
                    f"{QDRANT_URL}/collections/{collection}/points/payload",
                    data={"points": [point["id"]], "payload": {"semantic_links": links}},
                    method="POST",
                    timeout=10,
                )
                total_links += len(links)
        except Exception as e:
            if i < 3:
                print(f"    Link error for point {point['id']}: {e}")

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(points)} processed, {total_links} links created")

    print(f"  {collection}: {total_links} links created")
    return total_links


def main():
    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)

    conv_indices = list(range(len(conversations)))
    if len(sys.argv) > 1 and "--conversations" in sys.argv:
        idx = sys.argv.index("--conversations")
        conv_indices = [int(x) for x in sys.argv[idx + 1].split(",")]

    total_links = 0
    for idx in conv_indices:
        conv = conversations[idx]
        conv_id = conv.get("sample_id", f"conv-{idx}")
        collection = f"locomo_lb_{conv_id.replace('-', '_')}"
        total_links += create_links_for_collection(collection)

    print(f"\nTotal links created: {total_links}")


if __name__ == "__main__":
    main()
