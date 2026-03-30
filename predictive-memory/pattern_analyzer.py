#!/usr/bin/env python3
"""
Pattern Analyzer — Discovers temporal, sequential, and co-occurrence patterns
from memory access logs. Builds association maps for the anticipator.
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.path.expanduser("~/.openclaw/workspace/memory/predictive"))
ACCESS_LOG = DATA_DIR / "access_log.jsonl"
ASSOCIATIONS_FILE = DATA_DIR / "associations.json"
PATTERNS_FILE = DATA_DIR / "patterns.json"

# Half-life for exponential decay (days)
DECAY_HALF_LIFE = 7.0


def load_access_log() -> list[dict]:
    """Load all access log entries."""
    if not ACCESS_LOG.exists():
        return []
    entries = []
    with open(ACCESS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def compute_decay_score(ts_str: str) -> float:
    """Exponential decay score — recent = higher."""
    try:
        ts = datetime.fromisoformat(ts_str)
        age_days = (datetime.now() - ts).total_seconds() / 86400
        return math.exp(-0.693 * age_days / DECAY_HALF_LIFE)
    except (ValueError, TypeError):
        return 0.1


def build_co_occurrence_map(entries: list[dict]) -> dict[str, dict[str, float]]:
    """
    Build weighted co-occurrence map: topic A appears with topic B.
    Weights decay with age.
    """
    cooccur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    for entry in entries:
        topics = entry.get("topics", [])
        decay = compute_decay_score(entry.get("ts", ""))
        
        for i, t1 in enumerate(topics):
            for t2 in topics[i+1:]:
                cooccur[t1][t2] += decay
                cooccur[t2][t1] += decay
    
    # Normalize to 0-1
    result = {}
    for topic, related in cooccur.items():
        if not related:
            continue
        max_score = max(related.values()) if related else 1
        result[topic] = {k: round(v / max_score, 3) for k, v in 
                         sorted(related.items(), key=lambda x: -x[1])[:10]}
    
    return result


def build_sequential_patterns(entries: list[dict], window_minutes: int = 30) -> dict[str, list[str]]:
    """
    Find sequential patterns: topic A is often followed by topic B within N minutes.
    """
    sequences: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    sorted_entries = sorted(entries, key=lambda e: e.get("ts", ""))
    
    for i, entry in enumerate(sorted_entries):
        ts1 = entry.get("ts", "")
        topics1 = entry.get("topics", [])
        
        # Look ahead within window
        for j in range(i + 1, min(i + 20, len(sorted_entries))):
            ts2 = sorted_entries[j].get("ts", "")
            topics2 = sorted_entries[j].get("topics", [])
            
            try:
                dt1 = datetime.fromisoformat(ts1)
                dt2 = datetime.fromisoformat(ts2)
                if (dt2 - dt1).total_seconds() > window_minutes * 60:
                    break
            except (ValueError, TypeError):
                continue
            
            decay = compute_decay_score(ts1)
            for t1 in topics1:
                for t2 in topics2:
                    if t1 != t2:
                        sequences[t1][t2] += decay
    
    # Return top 5 followers per topic
    result = {}
    for topic, followers in sequences.items():
        sorted_f = sorted(followers.items(), key=lambda x: -x[1])[:5]
        result[topic] = [f[0] for f in sorted_f]
    
    return result


def build_temporal_patterns(entries: list[dict]) -> dict[int, list[str]]:
    """
    Discover which topics are queried at which hours.
    Returns {hour: [top_topics]} weighted by recency.
    """
    hourly: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    for entry in entries:
        h = entry.get("hour", 0)
        decay = compute_decay_score(entry.get("ts", ""))
        for topic in entry.get("topics", []):
            hourly[h][topic] += decay
    
    result = {}
    for h in range(24):
        if h in hourly:
            sorted_t = sorted(hourly[h].items(), key=lambda x: -x[1])[:5]
            result[h] = [t[0] for t in sorted_t]
    
    return result


def build_heat_scores(entries: list[dict]) -> dict[str, float]:
    """
    Compute heat score per topic: sum of decay-weighted accesses.
    Higher = hotter (more recently/frequently accessed).
    """
    heat: dict[str, float] = defaultdict(float)
    
    for entry in entries:
        decay = compute_decay_score(entry.get("ts", ""))
        for topic in entry.get("topics", []):
            heat[topic] += decay
    
    # Normalize to 0-100
    max_heat = max(heat.values()) if heat else 1
    return {k: round(v / max_heat * 100, 1) for k, v in 
            sorted(heat.items(), key=lambda x: -x[1])}


def analyze_and_save():
    """Run full analysis and save results."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_access_log()
    
    if not entries:
        print("No access log entries yet. Seeding with default associations.")
        # Seed with known associations from USER.md / entity_graph.json
        associations = {
            "family": {"household": 1.0, "planning": 0.9, "wellness": 0.8, "travel": 0.7, "documents": 0.6},
            "dad": {"transplant": 1.0, "ipf": 0.9, "toronto": 0.8, "medications": 0.7, "health": 0.6},
            "business": {"revenue": 1.0, "deposits": 0.9, "platform_a": 0.8, "platform_b": 0.7, "licensing": 0.6, "growth": 0.8, "curacao": 0.6},
            "health": {"testosterone": 1.0, "peptides": 0.9, "mounjaro": 0.8, "whoop": 0.7, "cgm": 0.6},
            "tech": {"server": 1.0, "gpu": 0.9, "ollama": 0.8, "qdrant": 0.7, "proxy": 0.6},
            "crypto": {"bitcoin": 1.0, "usdt": 0.9, "business": 0.5},
            "travel": {"passport": 1.0, "family": 0.8, "citizenship": 0.7},
            "cars": {"ferrari": 1.0, "supercar": 0.9, "gumball": 0.8},
        }
        patterns = {
            "temporal": {str(h): ["general"] for h in range(9, 23)},
            "sequential": {
                "family": ["planning", "health"],
                "business": ["business", "crypto"],
                "health": ["supplements", "family"],
            },
            "heat": {"business": 80, "health": 70, "finance": 70, "family": 60, "tech": 50},
        }
    else:
        print(f"Analyzing {len(entries)} access log entries...")
        associations = build_co_occurrence_map(entries)
        
        # Merge with seed associations (seeds have lower weight)
        seed = {
            "family": {"planning": 0.5, "medical": 0.4, "supplements": 0.3, "documents": 0.3},
            "dad": {"transplant": 0.5, "ipf": 0.4, "toronto": 0.3},
            "business": {"revenue": 0.5, "deposits": 0.4, "platform_a": 0.3},
            "health": {"testosterone": 0.5, "peptides": 0.4, "mounjaro": 0.3},
        }
        for topic, related in seed.items():
            if topic not in associations:
                associations[topic] = related
            else:
                for k, v in related.items():
                    if k not in associations[topic]:
                        associations[topic][k] = v
        
        temporal = build_temporal_patterns(entries)
        sequential = build_sequential_patterns(entries)
        heat = build_heat_scores(entries)
        
        patterns = {
            "temporal": temporal,
            "sequential": sequential,
            "heat": heat,
            "last_analyzed": datetime.now().isoformat(),
            "entries_analyzed": len(entries),
        }
    
    with open(ASSOCIATIONS_FILE, "w") as f:
        json.dump(associations, f, indent=2)
    
    with open(PATTERNS_FILE, "w") as f:
        json.dump(patterns, f, indent=2)
    
    print(f"Associations saved: {len(associations)} topics")
    print(f"Patterns saved to {PATTERNS_FILE}")
    return associations, patterns


if __name__ == "__main__":
    assoc, patt = analyze_and_save()
    print("\n--- Associations ---")
    for topic, related in list(assoc.items())[:5]:
        print(f"  {topic} → {list(related.keys())[:5]}")
    print("\n--- Heat Map ---")
    heat = patt.get("heat", {})
    for topic, score in list(heat.items())[:10]:
        bar = "█" * int(score / 5)
        print(f"  {topic:15s} {bar} {score}")
