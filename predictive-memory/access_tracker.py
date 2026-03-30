#!/usr/bin/env python3
"""
Access Tracker — Logs every memory access for pattern learning.
Append-only JSONL, lightweight, zero dependencies beyond stdlib.
"""

import json
import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("PREDICTIVE_DATA_DIR", "./data/memory/predictive"))
ACCESS_LOG = DATA_DIR / "access_log.jsonl"

# Common topic extraction patterns
ENTITY_KEYWORDS = {
    "family": ["family", "spouse", "partner", "household", "relatives"],
    "health": ["health", "medical", "doctor", "appointment", "wellness", "supplements"],
    "business": ["business", "revenue", "deposits", "metrics", "growth", "performance", "analytics", "licensing", "platform", "operations"],
    "crypto": ["cryptocurrency", "btc", "crypto"],
    "tech": ["ollama", "qdrant", "${WORKSPACE_NAME:-memory}", "server", "gpu", "vllm", "proxy"],
    "travel": ["passport", "travel", "visa", "citizenship"],
    "cars": ["ferrari", "supercar", "gumball"],
}


def extract_topics(query: str) -> list[str]:
    """Extract topic tags from a query string."""
    q = query.lower()
    topics = []
    for topic, keywords in ENTITY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            topics.append(topic)
    return topics or ["general"]


def log_access(query: str, results_count: int = 0, source: str = "recall", session_id: str = ""):
    """Log a memory access event."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "ts": datetime.now().isoformat(),
        "hour": datetime.now().hour,
        "weekday": datetime.now().weekday(),  # 0=Mon
        "query": query,
        "topics": extract_topics(query),
        "results": results_count,
        "source": source,
        "session": session_id,
    }
    
    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return entry


def get_recent_accesses(hours: int = 24) -> list[dict]:
    """Get access entries from the last N hours."""
    if not ACCESS_LOG.exists():
        return []
    
    cutoff = datetime.now().timestamp() - (hours * 3600)
    entries = []
    
    with open(ACCESS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["ts"]).timestamp()
                if ts >= cutoff:
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return entries


def get_topic_frequency(hours: int = 168) -> dict[str, int]:
    """Get topic access frequency over last N hours (default: 1 week)."""
    accesses = get_recent_accesses(hours)
    freq = {}
    for entry in accesses:
        for topic in entry.get("topics", []):
            freq[topic] = freq.get(topic, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))


def get_hourly_patterns() -> dict[int, list[str]]:
    """Discover which topics are queried at which hours."""
    accesses = get_recent_accesses(168)  # 1 week
    hourly: dict[int, dict[str, int]] = {}
    
    for entry in accesses:
        h = entry.get("hour", 0)
        if h not in hourly:
            hourly[h] = {}
        for topic in entry.get("topics", []):
            hourly[h][topic] = hourly[h].get(topic, 0) + 1
    
    # Return top 3 topics per hour
    result = {}
    for h, topics in sorted(hourly.items()):
        sorted_topics = sorted(topics.items(), key=lambda x: -x[1])
        result[h] = [t[0] for t in sorted_topics[:3]]
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        entry = log_access(query, source="cli_test")
        print(f"Logged: {json.dumps(entry, indent=2)}")
    else:
        print(f"Access log: {ACCESS_LOG}")
        print(f"Recent (24h): {len(get_recent_accesses(24))} entries")
        print(f"Topic frequency (7d): {get_topic_frequency()}")
        print(f"Hourly patterns: {json.dumps(get_hourly_patterns(), indent=2)}")
