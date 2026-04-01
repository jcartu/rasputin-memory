#!/usr/bin/env python3
"""RASPUTIN Memory Search Quality Benchmark — AFTER changes — hybrid_brain_v5"""

import json
import time
import urllib.request
import urllib.parse
import statistics
from datetime import datetime

BASE_URL = "http://localhost:7777"
OUTPUT_DIR = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/after"

QUERIES = {
    "personal": [
        "Josh's wife's name",
        "Josh's cats",
        "Where does Josh live",
        "Josh's brother Jonathan",
        "Josh's dad health",
        "Josh birthday",
        "Sasha birthday",
        "Josh's car interests",
        "Josh testosterone protocol",
        "Josh Whoop device",
    ],
    "business": [
        "WikiLuck revenue",
        "Curacao license renewal",
        "BetOBet brands",
        "DACH market percentage",
        "Rival platform casinos",
        "Delasport brand IDs",
        "ORBIT API",
        "casino payment processors",
        "NewEra BV license number",
        "monthly revenue",
    ],
    "technical": [
        "GPU setup RASPUTIN",
        "vLLM config",
        "Qdrant vectors count",
        "PM2 services list",
        "llama-swap routing",
        "cartu-proxy port",
        "nomic-embed-text model",
        "CUDA version",
        "RTX PRO 6000 specs",
        "Ollama port",
    ],
    "people": [
        "Alexander Durandin",
        "Jonathan Cartu wife",
        "Who is Sasha",
        "Lazar Cartu health",
        "David Cartu",
        "Teddy Sagi",
        "Nezemnoi birthday",
        "Mark birthday",
        "Vika",
        "Ashley",
    ],
    "temporal": [
        "recent war updates",
        "latest model benchmarks",
        "Houthi attacks",
        "oil price",
        "Iran situation",
    ],
    "edge_cases": [
        "BTC",
        "деньги",
        "Johs wife",
        "casino regulations requirements licensing jurisdictions operators must comply anti money laundering know your customer responsible gambling age verification payment processing fraud prevention data protection GDPR compliance technical standards RNG certification game fairness audit",
        "the thing",
    ],
}

def search(query, limit=10):
    encoded = urllib.parse.quote(query)
    url = f"{BASE_URL}/search?q={encoded}&limit={limit}"
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        elapsed_ms = (time.time() - start) * 1000
        return data, elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, str(e)

def judge_relevance(query, result_text):
    """Simple heuristic relevance judgment based on keyword overlap"""
    q_words = set(query.lower().split())
    t_words = set(result_text.lower().split())
    # Remove stopwords
    stops = {"the","a","an","of","in","is","are","was","were","to","for","and","or","with","that","this","has","have"}
    q_words -= stops
    t_words -= stops
    if not q_words:
        return 1
    overlap = len(q_words & t_words) / len(q_words)
    if overlap >= 0.5:
        return 3
    elif overlap >= 0.25:
        return 2
    elif overlap >= 0.1:
        return 1
    return 0

def calc_mrr(relevance_scores):
    for i, s in enumerate(relevance_scores):
        if s >= 2:
            return 1.0 / (i + 1)
    return 0.0

def calc_precision_at_k(relevance_scores, k):
    if not relevance_scores:
        return 0.0
    top_k = relevance_scores[:k]
    relevant = sum(1 for s in top_k if s >= 2)
    return relevant / k

results = []
latencies = []
all_categories = {}

print(f"Starting search benchmark at {datetime.now().isoformat()}")
print("="*60)

for category, queries in QUERIES.items():
    cat_results = []
    print(f"\nCategory: {category}")
    for query in queries:
        data, elapsed_ms, error = search(query, limit=10)
        latencies.append(elapsed_ms)
        
        entry = {
            "query": query,
            "category": category,
            "latency_ms": round(elapsed_ms, 2),
            "error": error,
            "num_results": 0,
            "results": [],
            "relevance_scores": [],
            "mrr": 0.0,
            "p_at_1": 0.0,
            "p_at_3": 0.0,
            "p_at_5": 0.0,
            "p_at_10": 0.0,
        }
        
        if data and not error:
            # Handle different response formats
            items = data if isinstance(data, list) else data.get("results", data.get("memories", []))
            entry["num_results"] = len(items)
            
            rel_scores = []
            for item in items[:10]:
                if isinstance(item, dict):
                    text = item.get("text", item.get("content", item.get("memory", "")))
                    score = item.get("score", item.get("relevance", 0))
                    source = item.get("source", item.get("metadata", {}).get("source", "unknown") if isinstance(item.get("metadata"), dict) else "unknown")
                else:
                    text = str(item)
                    score = 0
                    source = "unknown"
                
                rel = judge_relevance(query, text)
                rel_scores.append(rel)
                entry["results"].append({
                    "text_preview": text[:150],
                    "score": score,
                    "source": source,
                    "relevance": rel,
                })
            
            entry["relevance_scores"] = rel_scores
            entry["mrr"] = round(calc_mrr(rel_scores), 4)
            entry["p_at_1"] = round(calc_precision_at_k(rel_scores, 1), 4)
            entry["p_at_3"] = round(calc_precision_at_k(rel_scores, 3), 4)
            entry["p_at_5"] = round(calc_precision_at_k(rel_scores, 5), 4)
            entry["p_at_10"] = round(calc_precision_at_k(rel_scores, 10), 4)
        
        print(f"  [{elapsed_ms:.0f}ms] {query[:50]:<50} | n={entry['num_results']} | MRR={entry['mrr']:.2f} | P@1={entry['p_at_1']:.2f}")
        results.append(entry)
        cat_results.append(entry)
        time.sleep(0.05)  # small delay to avoid hammering
    
    all_categories[category] = cat_results

# Aggregate stats
valid_results = [r for r in results if not r["error"]]
if valid_results:
    mean_mrr = statistics.mean(r["mrr"] for r in valid_results)
    mean_p1 = statistics.mean(r["p_at_1"] for r in valid_results)
    mean_p3 = statistics.mean(r["p_at_3"] for r in valid_results)
    mean_p5 = statistics.mean(r["p_at_5"] for r in valid_results)
    mean_p10 = statistics.mean(r["p_at_10"] for r in valid_results)
    median_latency = statistics.median(latencies)
    sorted_lat = sorted(latencies)
    p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99_latency = sorted_lat[int(len(sorted_lat) * 0.99)]
else:
    mean_mrr = mean_p1 = mean_p3 = mean_p5 = mean_p10 = 0
    median_latency = p95_latency = p99_latency = 0

aggregate = {
    "total_queries": len(results),
    "successful_queries": len(valid_results),
    "mean_mrr": round(mean_mrr, 4),
    "mean_p_at_1": round(mean_p1, 4),
    "mean_p_at_3": round(mean_p3, 4),
    "mean_p_at_5": round(mean_p5, 4),
    "mean_p_at_10": round(mean_p10, 4),
    "median_latency_ms": round(median_latency, 2),
    "p95_latency_ms": round(p95_latency, 2),
    "p99_latency_ms": round(p99_latency, 2),
    "min_latency_ms": round(min(latencies), 2),
    "max_latency_ms": round(max(latencies), 2),
}

# Per-category aggregates
cat_aggregates = {}
for cat, cat_results in all_categories.items():
    valid = [r for r in cat_results if not r["error"]]
    if valid:
        cat_aggregates[cat] = {
            "n": len(valid),
            "mean_mrr": round(statistics.mean(r["mrr"] for r in valid), 4),
            "mean_p_at_1": round(statistics.mean(r["p_at_1"] for r in valid), 4),
            "mean_p_at_3": round(statistics.mean(r["p_at_3"] for r in valid), 4),
            "mean_p_at_5": round(statistics.mean(r["p_at_5"] for r in valid), 4),
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in valid), 2),
        }

output = {
    "benchmark_date": datetime.now().isoformat(),
    "aggregate": aggregate,
    "per_category": cat_aggregates,
    "queries": results,
}

out_path = f"{OUTPUT_DIR}/search_benchmark.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)
print(f"Total queries:    {aggregate['total_queries']}")
print(f"Successful:       {aggregate['successful_queries']}")
print(f"Mean MRR:         {aggregate['mean_mrr']:.4f}")
print(f"Mean P@1:         {aggregate['mean_p_at_1']:.4f}")
print(f"Mean P@3:         {aggregate['mean_p_at_3']:.4f}")
print(f"Mean P@5:         {aggregate['mean_p_at_5']:.4f}")
print(f"Mean P@10:        {aggregate['mean_p_at_10']:.4f}")
print(f"Median latency:   {aggregate['median_latency_ms']:.1f}ms")
print(f"P95 latency:      {aggregate['p95_latency_ms']:.1f}ms")
print(f"P99 latency:      {aggregate['p99_latency_ms']:.1f}ms")
print("\nPer-category MRR:")
for cat, stats in cat_aggregates.items():
    print(f"  {cat:<15} MRR={stats['mean_mrr']:.3f}  P@1={stats['mean_p_at_1']:.3f}  lat={stats['mean_latency_ms']:.0f}ms")
print(f"\nSaved to {out_path}")
