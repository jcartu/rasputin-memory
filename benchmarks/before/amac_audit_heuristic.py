#!/usr/bin/env python3
"""A-MAC Quality Audit — Heuristic version (LLM scoring not feasible due to thinking model overhead).
Uses payload-based heuristics to estimate quality scores.
Note: Full LLM audit at 3+ sec/call * 500 samples = 25+ min. This heuristic runs in seconds."""

import json
import urllib.request
import random
import statistics
from datetime import datetime

QDRANT = "http://localhost:6333"
OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/amac_audit_500.jsonl"
STATS_OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/amac_stats.json"

SAMPLE_SIZE = 500
AMAC_THRESHOLD = 4.0

# Source quality weights (based on known source reliability)
SOURCE_WEIGHTS = {
    "telegram": 0.7,
    "perplexity": 0.8,
    "matrix": 0.65,
    "swarm-vanguard": 0.75,
    "conversation": 0.7,
    "swarm-scout": 0.72,
    "consolidator-v4": 0.85,
    "rd-scan": 0.80,
    "memory-consolidator-v2": 0.80,
    "chatgpt": 0.7,
    "fact-extractor": 0.90,
    "manual": 0.95,
    "windows_chrome": 0.6,
    "daily-retro": 0.75,
    "session-digest": 0.75,
    "benchmark_test": 0.0,
}

def heuristic_score(payload, text):
    """Score memory using heuristics."""
    text_len = len(text)
    source = payload.get("source", "unknown")
    importance = float(payload.get("importance", payload.get("importance_score", 50)) or 50)
    
    # Relevance: based on importance score and source quality
    src_weight = SOURCE_WEIGHTS.get(source, 0.65)
    R = (importance / 100.0) * 10 * src_weight
    R = max(0, min(10, R))
    
    # Novelty: based on text length and content diversity
    # Longer, more detailed memories are more likely to be novel
    if text_len < 30:
        N = 2.0
    elif text_len < 100:
        N = 4.0
    elif text_len < 300:
        N = 6.0
    elif text_len < 800:
        N = 7.5
    else:
        N = 8.5
    # Penalize obvious duplicates by common phrases
    dup_phrases = ["test memory", "benchmark", "lorem ipsum"]
    if any(p in text.lower() for p in dup_phrases):
        N = 1.0
    
    # Specificity: based on presence of specific entities (numbers, names, dates, URLs)
    import re
    has_numbers = bool(re.search(r'\d+', text))
    has_names = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text))
    has_dates = bool(re.search(r'\d{4}[-/]\d{1,2}', text))
    has_urls = bool(re.search(r'https?://', text))
    specificity_score = 4.0
    if has_numbers: specificity_score += 1.5
    if has_names: specificity_score += 1.5
    if has_dates: specificity_score += 1.5
    if has_urls: specificity_score += 1.0
    specificity_score = min(10, specificity_score)
    S = specificity_score
    
    composite = (R + N + S) / 3.0
    return round(R, 2), round(N, 2), round(S, 2), round(composite, 2)

def qdrant_post(path, body):
    url = f"{QDRANT}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

print(f"Starting A-MAC heuristic audit at {datetime.now().isoformat()}")

# Scroll through and collect samples
print(f"Sampling {SAMPLE_SIZE} memories via scroll...")
all_points = []
offset = None
while len(all_points) < SAMPLE_SIZE * 3:
    body = {"limit": 250, "with_payload": True, "with_vector": False}
    if offset:
        body["offset"] = offset
    resp = qdrant_post("/collections/memories_v2/points/scroll", body)
    pts = resp["result"]["points"]
    if not pts:
        break
    all_points.extend(pts)
    offset = resp["result"].get("next_page_offset")
    if not offset or len(all_points) >= 3000:
        break

random.shuffle(all_points)
sample_points = all_points[:SAMPLE_SIZE]
print(f"Got {len(sample_points)} samples")

results = []
with open(OUTPUT, "w") as out_f:
    for p in sample_points:
        payload = p.get("payload", {})
        text = payload.get("text", payload.get("content", ""))
        if not text:
            continue
        source = payload.get("source", "unknown")
        date = payload.get("date", payload.get("created_at", ""))
        importance = payload.get("importance", payload.get("importance_score", 0))
        
        R, N, S, composite = heuristic_score(payload, text)
        
        entry = {
            "point_id": p["id"],
            "text_preview": text[:150],
            "source": source,
            "date": date,
            "importance": importance,
            "R": R,
            "N": N,
            "S": S,
            "composite": composite,
            "would_reject": composite < AMAC_THRESHOLD,
            "method": "heuristic",
        }
        out_f.write(json.dumps(entry) + "\n")
        results.append(entry)

print(f"Scored {len(results)} memories")

if results:
    composites = [r["composite"] for r in results]
    r_scores = [r["R"] for r in results]
    n_scores = [r["N"] for r in results]
    s_scores = [r["S"] for r in results]
    would_reject = sum(1 for r in results if r["would_reject"])
    
    hist = {f"{i}-{i+1}": 0 for i in range(0, 10)}
    for c in composites:
        bucket = min(int(c), 9)
        hist[f"{bucket}-{bucket+1}"] += 1
    
    # By source quality
    source_stats = {}
    for r in results:
        src = r["source"]
        if src not in source_stats:
            source_stats[src] = []
        source_stats[src].append(r["composite"])
    source_summary = {src: round(statistics.mean(scores), 2) for src, scores in source_stats.items() if scores}
    
    stats = {
        "benchmark_date": datetime.now().isoformat(),
        "method": "heuristic (LLM scoring not feasible - qwen3 thinking model overhead ~3s/call)",
        "sample_size": len(results),
        "threshold": AMAC_THRESHOLD,
        "composite_scores": {
            "mean": round(statistics.mean(composites), 3),
            "median": round(statistics.median(composites), 3),
            "stdev": round(statistics.stdev(composites) if len(composites) > 1 else 0, 3),
            "min": round(min(composites), 3),
            "max": round(max(composites), 3),
        },
        "dimension_means": {
            "R_relevance": round(statistics.mean(r_scores), 3),
            "N_novelty": round(statistics.mean(n_scores), 3),
            "S_specificity": round(statistics.mean(s_scores), 3),
        },
        "rejection_rate_at_4": round(would_reject / len(results), 4),
        "would_reject_count": would_reject,
        "score_histogram": hist,
        "mean_score_by_source": dict(sorted(source_summary.items(), key=lambda x: -x[1])),
    }
    
    with open(STATS_OUTPUT, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\nA-MAC Heuristic Results:")
    print(f"  Mean composite:  {stats['composite_scores']['mean']:.2f}")
    print(f"  Median:          {stats['composite_scores']['median']:.2f}")
    print(f"  Stdev:           {stats['composite_scores']['stdev']:.2f}")
    print(f"  Reject rate @{AMAC_THRESHOLD}: {stats['rejection_rate_at_4']:.1%} ({would_reject}/{len(results)})")
    print(f"\n  Dimensions: R={stats['dimension_means']['R_relevance']:.2f}  N={stats['dimension_means']['N_novelty']:.2f}  S={stats['dimension_means']['S_specificity']:.2f}")
    print("\n  Score histogram:")
    for k, v in hist.items():
        bar = "█" * (v // 5)
        print(f"    {k}: {v:>4} {bar}")
    print("\n  Mean score by source:")
    for src, score in list(source_summary.items())[:10]:
        print(f"    {src:<30} {score:.2f}")
    print(f"\nSaved to {OUTPUT} and {STATS_OUTPUT}")
