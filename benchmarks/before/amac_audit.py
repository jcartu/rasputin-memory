#!/usr/bin/env python3
"""A-MAC Quality Audit — BEFORE changes. Samples 100 random memories and scores them."""

import json
import urllib.request
import random
import statistics
import time
from datetime import datetime

QDRANT = "http://localhost:6333"
LLM_URL = "http://localhost:11435/v1/chat/completions"
OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/amac_audit_500.jsonl"
STATS_OUTPUT = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/before/amac_stats.json"

SAMPLE_SIZE = 100  # Reduced from 500 for time; extrapolate
AMAC_THRESHOLD = 4.0

AMAC_PROMPT = """Score this memory on three dimensions. Reply with ONLY a JSON object, nothing else.

Memory: {text}

Score each dimension 0-10:
- R (Relevance): How relevant is this to building a useful personal knowledge base? (0=useless noise, 10=highly valuable personal/professional info)
- N (Novelty): How unique/novel is this? Would removing it cause significant information loss? (0=completely redundant, 10=unique insight)  
- S (Specificity): How specific and concrete is this? (0=vague generality, 10=precise actionable detail)

Reply format: {{"R": <0-10>, "N": <0-10>, "S": <0-10>, "composite": <average>}}"""

def qdrant_post(path, body):
    url = f"{QDRANT}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def score_memory(text):
    prompt = AMAC_PROMPT.format(text=text[:500])
    body = {
        "model": "qwen3.5-122b-a10b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(LLM_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"].strip()
    # Parse JSON from content
    # Remove thinking tags if present
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    # Find JSON
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        scores = json.loads(content[start:end])
        r_score = float(scores.get("R", 5))
        n_score = float(scores.get("N", 5))
        s_score = float(scores.get("S", 5))
        composite = scores.get("composite", (r_score + n_score + s_score) / 3)
        return r_score, n_score, s_score, float(composite)
    return None

print(f"Starting A-MAC audit at {datetime.now().isoformat()}")
print(f"Sampling {SAMPLE_SIZE} memories...")

# Collect point IDs
all_ids = []
offset = None
while len(all_ids) < SAMPLE_SIZE * 5:
    body = {"limit": 250, "with_payload": False, "with_vector": False}
    if offset:
        body["offset"] = offset
    resp = qdrant_post("/collections/memories_v2/points/scroll", body)
    pts = resp["result"]["points"]
    if not pts:
        break
    all_ids.extend(p["id"] for p in pts)
    offset = resp["result"].get("next_page_offset")
    if not offset or len(all_ids) >= 5000:
        break

random.shuffle(all_ids)
sample_ids = all_ids[:SAMPLE_SIZE]
print(f"Got {len(sample_ids)} sample IDs")

results = []
errors = 0

with open(OUTPUT, "w") as out_f:
    for i, pt_id in enumerate(sample_ids):
        if i % 10 == 0:
            print(f"  {i}/{SAMPLE_SIZE}...")
        
        try:
            # Get payload
            resp = qdrant_post("/collections/memories_v2/points", {"ids": [pt_id], "with_vector": False, "with_payload": True})
            if not resp["result"]:
                continue
            point = resp["result"][0]
            payload = point.get("payload", {})
            text = payload.get("text", payload.get("content", ""))
            source = payload.get("source", "unknown")
            date = payload.get("date", payload.get("created_at", ""))
            importance = payload.get("importance", payload.get("importance_score", 0))
            
            if not text or len(text) < 10:
                continue
            
            # Score
            scores = score_memory(text)
            if scores is None:
                errors += 1
                continue
            
            r_score, n_score, s_score, composite = scores
            
            entry = {
                "point_id": pt_id,
                "text_preview": text[:150],
                "source": source,
                "date": date,
                "importance": importance,
                "R": r_score,
                "N": n_score,
                "S": s_score,
                "composite": round(composite, 2),
                "would_reject": composite < AMAC_THRESHOLD,
            }
            out_f.write(json.dumps(entry) + "\n")
            results.append(entry)
        except Exception as e:
            errors += 1
            if i < 5:
                print(f"  Error on {pt_id}: {e}")
        
        time.sleep(0.05)

print(f"\nScored {len(results)} memories, {errors} errors")

if results:
    composites = [r["composite"] for r in results]
    r_scores = [r["R"] for r in results]
    n_scores = [r["N"] for r in results]
    s_scores = [r["S"] for r in results]
    would_reject = sum(1 for r in results if r["would_reject"])
    
    # Histogram
    hist = {f"{i}-{i+1}": 0 for i in range(0, 10)}
    for c in composites:
        bucket = min(int(c), 9)
        hist[f"{bucket}-{bucket+1}"] += 1
    
    stats = {
        "benchmark_date": datetime.now().isoformat(),
        "sample_size": len(results),
        "errors": errors,
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
    }
    
    with open(STATS_OUTPUT, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\nA-MAC Results:")
    print(f"  Mean composite: {stats['composite_scores']['mean']:.2f}")
    print(f"  Median:         {stats['composite_scores']['median']:.2f}")
    print(f"  Stdev:          {stats['composite_scores']['stdev']:.2f}")
    print(f"  Rejection rate (< {AMAC_THRESHOLD}): {stats['rejection_rate_at_4']:.1%}")
    print(f"  Would reject:   {would_reject}/{len(results)}")
    print(f"\n  Dimension means: R={stats['dimension_means']['R_relevance']:.2f}  N={stats['dimension_means']['N_novelty']:.2f}  S={stats['dimension_means']['S_specificity']:.2f}")
    print("\n  Histogram:")
    for k, v in hist.items():
        bar = "█" * (v // 1)
        print(f"    {k}: {v:>4} {bar}")
    
    print(f"\nSaved to {OUTPUT}")
    print(f"Saved stats to {STATS_OUTPUT}")
