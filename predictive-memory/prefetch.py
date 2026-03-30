#!/usr/bin/env python3
"""
Pre-fetch Daemon — Runs on cron to populate the predictive cache.
Queries Qdrant for predicted-to-be-needed memories and caches results.

Usage:
  python3 prefetch.py --morning    # Heavy morning pre-fetch (8am cron)
  python3 prefetch.py --refresh    # Light refresh (every 2h cron)
  python3 prefetch.py --topic X    # Pre-fetch specific topic
"""

import json
import os
import sys
import requests
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("PREDICTIVE_DATA_DIR", "./data/memory/predictive"))
CACHE_FILE = DATA_DIR / "cache.json"
QDRANT_SEARCH = "${MEMORY_API_URL:-http://${MEMORY_API_HOST:-localhost:7777}}/search"

# Queries per topic for pre-fetching
TOPIC_QUERIES = {
    "business": ["business revenue this month", "deposit numbers", "platform A performance", "platform B stats", "monthly revenue growth", "curacao license status", "platform performance"],
    "health": ["health updates", "medical status", "treatment progress", "wellness"],
    "family": ["family planning progress", "wellness supplements", "document status"],
    "tech": ["server status", "gpu utilization", "ollama performance", "qdrant stats"],
    "crypto": ["cryptocurrency price", "crypto operations", "crypto holdings"],
    "travel": ["passport application", "russian travel restrictions"],
    "cars": ["vehicle financing", "car maintenance"],
}

DEFAULT_TTL_HOURS = 4
MORNING_TTL_HOURS = 8


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic write
    tmp = CACHE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    tmp.rename(CACHE_FILE)


def search_qdrant(query: str, limit: int = 3) -> list[dict]:
    """Query Qdrant second brain."""
    try:
        resp = requests.get(QDRANT_SEARCH, params={"q": query, "limit": limit}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            # Slim down — only keep text and score
            return [{"text": r.get("text", "")[:500], "score": r.get("score", 0)} 
                    for r in results if r.get("score", 0) > 0.3]
    except Exception as e:
        print(f"  Qdrant error for '{query}': {e}")
    return []


def prefetch_topic(topic: str, ttl_hours: int = DEFAULT_TTL_HOURS) -> dict:
    """Pre-fetch all queries for a topic."""
    queries = TOPIC_QUERIES.get(topic, [topic])
    all_results = []
    
    for query in queries:
        results = search_qdrant(query, limit=3)
        if results:
            all_results.extend(results)
    
    # Deduplicate by text hash
    seen = set()
    unique = []
    for r in all_results:
        h = hash(r["text"][:100])
        if h not in seen:
            seen.add(h)
            unique.append(r)
    
    return {
        "topic": topic,
        "results": unique[:10],  # Max 10 per topic
        "queries_run": len(queries),
        "cached_at": datetime.now().isoformat(),
        "ttl_hours": ttl_hours,
    }


def run_morning_prefetch():
    """Heavy morning pre-fetch — all routine topics + hot from yesterday."""
    print(f"[{datetime.now().isoformat()}] Morning pre-fetch starting...")
    
    # Import pattern analyzer
    sys.path.insert(0, str(Path(__file__).parent))
    from anticipator import anticipate_for_briefing
    
    briefing = anticipate_for_briefing()
    topics = briefing.get("briefing_topics", ["business", "business", "health"])
    
    cache = load_cache()
    total_results = 0
    
    for topic in topics:
        print(f"  Pre-fetching: {topic}...")
        entry = prefetch_topic(topic, ttl_hours=MORNING_TTL_HOURS)
        cache[topic] = entry
        total_results += len(entry["results"])
    
    save_cache(cache)
    print(f"Morning pre-fetch complete: {len(topics)} topics, {total_results} cached results")
    return cache


def run_refresh():
    """Light refresh — only update expired entries and hot topics."""
    print(f"[{datetime.now().isoformat()}] Refresh pre-fetch starting...")
    
    cache = load_cache()
    refreshed = 0
    
    for topic, entry in list(cache.items()):
        cached_at = entry.get("cached_at", "")
        ttl = entry.get("ttl_hours", DEFAULT_TTL_HOURS)
        
        try:
            ct = datetime.fromisoformat(cached_at)
            age_hours = (datetime.now() - ct).total_seconds() / 3600
            if age_hours >= ttl:
                print(f"  Refreshing expired: {topic} (age: {age_hours:.1f}h)")
                cache[topic] = prefetch_topic(topic, ttl_hours=DEFAULT_TTL_HOURS)
                refreshed += 1
        except (ValueError, TypeError):
            cache[topic] = prefetch_topic(topic, ttl_hours=DEFAULT_TTL_HOURS)
            refreshed += 1
    
    save_cache(cache)
    print(f"Refresh complete: {refreshed} topics refreshed")
    return cache


def prefetch_specific(topic: str):
    """Pre-fetch a specific topic on demand."""
    cache = load_cache()
    entry = prefetch_topic(topic)
    cache[topic] = entry
    save_cache(cache)
    print(f"Pre-fetched '{topic}': {len(entry['results'])} results cached")
    return entry


if __name__ == "__main__":
    if "--morning" in sys.argv:
        run_morning_prefetch()
    elif "--refresh" in sys.argv:
        run_refresh()
    elif "--topic" in sys.argv:
        idx = sys.argv.index("--topic")
        if idx + 1 < len(sys.argv):
            prefetch_specific(sys.argv[idx + 1])
        else:
            print("Usage: --topic <topic_name>")
    elif "--status" in sys.argv:
        cache = load_cache()
        print(f"Cache: {len(cache)} topics")
        for topic, entry in cache.items():
            age = "?"
            try:
                ct = datetime.fromisoformat(entry.get("cached_at", ""))
                age = f"{(datetime.now() - ct).total_seconds() / 3600:.1f}h"
            except Exception:
                pass
            print(f"  {topic}: {len(entry.get('results', []))} results, age={age}, ttl={entry.get('ttl_hours', '?')}h")
    else:
        print("Usage: python3 prefetch.py [--morning|--refresh|--topic X|--status]")
