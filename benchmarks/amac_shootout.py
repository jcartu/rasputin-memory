#!/usr/bin/env python3
"""A-MAC Shootout: 3-way comparison of memory quality evaluation models."""

import requests
import json
import time
import re
import statistics
import math
from pathlib import Path

# Paths
RESULTS_PATH = Path("/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/amac_shootout_results.jsonl")
SUMMARY_PATH = Path("/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/amac_shootout_summary.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Variant configs
VARIANT_A = {
    "name": "35B Generic",
    "url": "http://localhost:11436/v1/chat/completions",
    "model": "qwen3.5:35b",
    "system": 'You are a memory quality evaluator. Score this memory on three criteria (0-10): Relevance (useful for future retrieval?), Novelty (genuinely new info?), Specificity (concrete retrievable details?). Return ONLY JSON: {"relevance":N,"novelty":N,"specificity":N,"composite":N} where composite = (R+N+S)/3. No explanation.',
    "max_tokens": 150,
}
VARIANT_B = {
    "name": "122B Generic",
    "url": "http://localhost:11435/v1/chat/completions",
    "model": "qwen3.5-122b-a10b",
    "system": 'You are a memory quality evaluator. Score this memory on three criteria (0-10): Relevance (useful for future retrieval?), Novelty (genuinely new info?), Specificity (concrete retrievable details?). Return ONLY JSON: {"relevance":N,"novelty":N,"specificity":N,"composite":N} where composite = (R+N+S)/3. No explanation.',
    "max_tokens": 150,
}
VARIANT_C = {
    "name": "122B Domain-Specific",
    "url": "http://localhost:11435/v1/chat/completions",
    "model": "qwen3.5-122b-a10b",
    "system": 'You are a memory quality gate for an AI agent serving a CEO running an iGaming business ($2.5M/mo revenue), tracking geopolitics, managing AI infrastructure (dual RTX PRO 6000 + RTX 5090), and personal/family context. Score on 5 criteria (0-10): RELEVANCE (future retrieval value: 0=filler, 5=useful, 10=critical), NOVELTY (new info: 0=obvious, 10=first mention), SPECIFICITY (concrete details: 0=vague, 10=self-contained fact), ACTIONABILITY (future need: 0=expired, 10=critical deadline), TEMPORAL (longevity: 0=expired, 10=permanent). COMPOSITE = R*0.30 + N*0.20 + S*0.20 + A*0.15 + T*0.15. Return ONLY JSON: {"relevance":N,"novelty":N,"specificity":N,"actionability":N,"temporal":N,"composite":N}',
    "max_tokens": 200,
}

def get_memories():
    print("Fetching 200 memories from Qdrant...")
    resp = requests.post(
        "http://localhost:6333/collections/memories_v2/points/scroll",
        json={"limit": 200, "with_payload": True, "with_vector": False},
        timeout=30
    )
    resp.raise_for_status()
    points = resp.json()["result"]["points"]
    print(f"Got {len(points)} points")
    return points

def extract_text(point):
    payload = point.get("payload", {})
    # Try common text fields
    for field in ["text", "content", "memory", "body", "value"]:
        if field in payload and payload[field]:
            return str(payload[field])
    # Fallback: join all string values
    parts = [str(v) for v in payload.values() if isinstance(v, str) and v]
    return " | ".join(parts) if parts else str(payload)

def parse_json_response(text):
    """Try to extract JSON from model response."""
    # Direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    # Find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    # Extract key:value pairs
    result = {}
    for key in ["relevance", "novelty", "specificity", "actionability", "temporal", "composite"]:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', text)
        if m:
            result[key] = float(m.group(1))
    if result:
        return result
    return None

def call_model(variant, memory_text):
    user_msg = f"Memory text: {memory_text[:1000]} /no_think"
    payload = {
        "model": variant["model"],
        "messages": [
            {"role": "system", "content": variant["system"]},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.1,
        "max_tokens": variant["max_tokens"],
    }
    start = time.time()
    try:
        resp = requests.post(variant["url"], json=payload, timeout=120)
        resp.raise_for_status()
        elapsed_ms = (time.time() - start) * 1000
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        return parsed, elapsed_ms, content
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        print(f"  ERROR calling {variant['name']}: {e}")
        return None, elapsed_ms, str(e)

def main():
    points = get_memories()
    
    results = []
    stats = {
        "a": {"composites": [], "latencies": [], "failures": 0},
        "b": {"composites": [], "latencies": [], "failures": 0},
        "c": {"composites": [], "latencies": [], "failures": 0},
    }
    
    print(f"\nProcessing {len(points)} memories sequentially...")
    
    with open(RESULTS_PATH, "w") as f_out:
        for i, point in enumerate(points):
            pid = point.get("id", i)
            text = extract_text(point)
            print(f"[{i+1}/{len(points)}] id={pid} text_len={len(text)}")
            
            record = {
                "id": pid,
                "text_preview": text[:120],
                "variants": {}
            }
            
            for key, variant in [("a", VARIANT_A), ("b", VARIANT_B), ("c", VARIANT_C)]:
                parsed, latency_ms, raw = call_model(variant, text)
                composite = parsed.get("composite") if parsed else None
                stats[key]["latencies"].append(latency_ms)
                if composite is not None:
                    stats[key]["composites"].append(float(composite))
                else:
                    stats[key]["failures"] += 1
                
                record["variants"][key] = {
                    "name": variant["name"],
                    "scores": parsed,
                    "composite": composite,
                    "latency_ms": round(latency_ms, 1),
                    "parse_ok": parsed is not None,
                }
                print(f"  {variant['name']}: composite={composite} latency={latency_ms:.0f}ms")
            
            results.append(record)
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
    
    print("\nComputing summary...")
    
    def variant_summary(key):
        c = stats[key]["composites"]
        l = stats[key]["latencies"]
        if not c:
            return {"mean_composite": None, "median": None, "stdev": None, "reject_rate_at_4": None, "mean_latency_ms": None, "parse_failures": stats[key]["failures"]}
        return {
            "mean_composite": round(statistics.mean(c), 3),
            "median": round(statistics.median(c), 3),
            "stdev": round(statistics.stdev(c) if len(c) > 1 else 0, 3),
            "reject_rate_at_4": round(sum(1 for x in c if x < 4) / len(c), 3),
            "mean_latency_ms": round(statistics.mean(l), 1),
            "parse_failures": stats[key]["failures"],
        }
    
    def correlation(xs, ys):
        """Pearson correlation between two lists (aligned by index)."""
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) < 2:
            return None
        len(pairs)
        xs2 = [p[0] for p in pairs]
        ys2 = [p[1] for p in pairs]
        mx, my = statistics.mean(xs2), statistics.mean(ys2)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den = math.sqrt(sum((x - mx)**2 for x in xs2) * sum((y - my)**2 for y in ys2))
        return round(num / den, 4) if den else None
    
    # Build composite lists aligned
    a_scores = [r["variants"]["a"]["composite"] for r in results]
    b_scores = [r["variants"]["b"]["composite"] for r in results]
    c_scores = [r["variants"]["c"]["composite"] for r in results]
    
    # Find interesting disagreements (largest A vs C delta)
    disagreements = []
    for r in results:
        a = r["variants"]["a"]["composite"]
        c = r["variants"]["c"]["composite"]
        b = r["variants"]["b"]["composite"]
        if a is not None and c is not None:
            delta = abs(a - c)
            disagreements.append((delta, r["text_preview"], a, b, c))
    disagreements.sort(reverse=True)
    top_disagreements = [
        {"memory_preview": d[1], "a_score": d[2], "b_score": d[3], "c_score": d[4]}
        for d in disagreements[:5]
    ]
    
    summary = {
        "total_memories": len(points),
        "variant_a_35b": variant_summary("a"),
        "variant_b_122b": variant_summary("b"),
        "variant_c_122b_domain": variant_summary("c"),
        "agreement": {
            "a_vs_b_correlation": correlation(a_scores, b_scores),
            "b_vs_c_correlation": correlation(b_scores, c_scores),
            "a_vs_c_correlation": correlation(a_scores, c_scores),
        },
        "interesting_disagreements": top_disagreements,
    }
    
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Done!")
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))
    
    return summary

if __name__ == "__main__":
    main()
