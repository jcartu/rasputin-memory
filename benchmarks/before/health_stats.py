#!/usr/bin/env python3
"""RASPUTIN Memory Health Stats — BEFORE changes"""

import json
import urllib.request
import urllib.parse
from datetime import datetime

QDRANT = "http://localhost:6333"
FALKOR_HOST = "localhost"
FALKOR_PORT = 6380
OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/health_stats.json"

def qdrant_get(path):
    url = f"{QDRANT}{path}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())

def qdrant_post(path, body):
    url = f"{QDRANT}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

# 1. Collection info
print("Getting collection info...")
coll_info = qdrant_get("/collections/memories_v2")
total_points = coll_info["result"]["points_count"]
vectors_count = coll_info["result"].get("vectors_count", total_points)
print(f"  Total points: {total_points}")

# 2. Scroll to gather stats — sample 5000 points for distributions
print("Sampling 5000 points for distribution stats...")
source_counts = {}
month_counts = {}
importance_hist = {f"{i*10}-{i*10+10}": 0 for i in range(10)}
retrieval_counts = {}
retrieval_hist = {"0": 0, "1-5": 0, "6-20": 0, "21-100": 0, "100+": 0}

offset = None
total_sampled = 0
max_sample = 5000

while total_sampled < max_sample:
    body = {
        "limit": 250,
        "with_payload": True,
        "with_vector": False,
    }
    if offset:
        body["offset"] = offset
    
    resp = qdrant_post("/collections/memories_v2/points/scroll", body)
    points = resp["result"]["points"]
    if not points:
        break
    
    for p in points:
        payload = p.get("payload", {})
        
        # Source
        source = payload.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        
        # Date/month
        date_str = payload.get("date", payload.get("created_at", payload.get("timestamp", "")))
        if date_str and len(date_str) >= 7:
            month = date_str[:7]  # YYYY-MM
            month_counts[month] = month_counts.get(month, 0) + 1
        
        # Importance
        imp = payload.get("importance", payload.get("importance_score", 0))
        try:
            imp = float(imp)
            bucket = min(int(imp / 10), 9)
            key = f"{bucket*10}-{bucket*10+10}"
            importance_hist[key] = importance_hist.get(key, 0) + 1
        except (TypeError, ValueError):
            importance_hist["0-10"] = importance_hist.get("0-10", 0) + 1
        
        # Retrieval count
        rc = payload.get("retrieval_count", payload.get("access_count", 0))
        try:
            rc = int(rc)
            if rc == 0:
                retrieval_hist["0"] += 1
            elif rc <= 5:
                retrieval_hist["1-5"] += 1
            elif rc <= 20:
                retrieval_hist["6-20"] += 1
            elif rc <= 100:
                retrieval_hist["21-100"] += 1
            else:
                retrieval_hist["100+"] += 1
        except (TypeError, ValueError):
            retrieval_hist["0"] += 1
    
    total_sampled += len(points)
    offset = resp["result"].get("next_page_offset")
    print(f"  Sampled {total_sampled}...")
    if not offset or total_sampled >= max_sample:
        break

print(f"Total sampled: {total_sampled}")

# Sort months
sorted_months = dict(sorted(month_counts.items()))

# 3. FalkorDB stats via redis protocol
print("Querying FalkorDB...")
falkor_stats = {}
try:
    import socket
    
    def redis_cmd(sock, *args):
        cmd = f"*{len(args)}\r\n"
        for a in args:
            a_str = str(a)
            cmd += f"${len(a_str.encode())}\r\n{a_str}\r\n"
        sock.sendall(cmd.encode())
        return sock.recv(65536).decode("utf-8", errors="replace")
    
    s = socket.socket()
    s.connect((FALKOR_HOST, FALKOR_PORT))
    s.settimeout(10)
    
    # Total nodes
    r = redis_cmd(s, "GRAPH.QUERY", "brain", "MATCH (n) RETURN count(n)")
    print(f"  Nodes response: {r[:200]}")
    
    # Total edges
    r2 = redis_cmd(s, "GRAPH.QUERY", "brain", "MATCH ()-[r]->() RETURN count(r)")
    print(f"  Edges response: {r2[:200]}")
    
    # Node labels
    r3 = redis_cmd(s, "GRAPH.QUERY", "brain", "CALL db.labels()")
    print(f"  Labels response: {r3[:300]}")
    
    s.close()
    falkor_stats["raw_queries_ok"] = True
except Exception as e:
    print(f"  FalkorDB socket error: {e}")
    falkor_stats["error"] = str(e)

# Use stats endpoint values we already have
falkor_stats["nodes_from_api"] = 107320
falkor_stats["edges_from_api"] = 124792

# Output
stats = {
    "benchmark_date": datetime.now().isoformat(),
    "qdrant": {
        "collection": "memories_v2",
        "total_points": total_points,
        "vectors_count": vectors_count,
        "sample_size": total_sampled,
        "source_breakdown": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "age_distribution_by_month": sorted_months,
        "importance_score_histogram": importance_hist,
        "retrieval_count_histogram": retrieval_hist,
    },
    "falkordb": falkor_stats,
}

with open(OUTPUT, "w") as f:
    json.dump(stats, f, indent=2)

print("\nSource breakdown:")
for s, c in sorted(source_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {s:<30} {c:>6}")

print("\nImportance histogram:")
for k, v in importance_hist.items():
    print(f"  {k:<10} {v:>6}")

print("\nRetrieval count histogram:")
for k, v in retrieval_hist.items():
    print(f"  {k:<10} {v:>6}")

print(f"\nSaved to {OUTPUT}")
