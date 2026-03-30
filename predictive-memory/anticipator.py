#!/usr/bin/env python3
"""
Context Anticipator — Given a topic or query, predict what information 
will be needed next. Sub-10ms response time (pure dict lookups).

Usage:
  python3 anticipator.py "family"
  python3 anticipator.py --topic business
  
  # From Python:
  from anticipator import anticipate
  predictions = anticipate("family mentioned appointment")
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from access_tracker import extract_topics
except ImportError:
    def extract_topics(text: str) -> list:
        """Fallback topic extractor."""
        return []

DATA_DIR = Path(os.path.expanduser("~/.openclaw/workspace/memory/predictive"))
ASSOCIATIONS_FILE = DATA_DIR / "associations.json"
PATTERNS_FILE = DATA_DIR / "patterns.json"
CACHE_FILE = DATA_DIR / "cache.json"


def load_associations() -> dict:
    """Load association map."""
    if ASSOCIATIONS_FILE.exists():
        with open(ASSOCIATIONS_FILE) as f:
            return json.load(f)
    return {}


def load_patterns() -> dict:
    """Load temporal/sequential patterns."""
    if PATTERNS_FILE.exists():
        with open(PATTERNS_FILE) as f:
            return json.load(f)
    return {}


def load_cache() -> dict:
    """Load pre-fetch cache."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def anticipate(query: str, max_predictions: int = 5) -> dict:
    """
    Given a query/topic, predict what information will be needed next.
    
    Returns:
        {
            "predicted_topics": ["fertility", "ivf", ...],
            "cached_results": [{"query": "...", "results": [...]}],
            "confidence": 0.0-1.0,
            "reasoning": "Based on X pattern..."
        }
    """
    start = time.monotonic()
    
    associations = load_associations()
    patterns = load_patterns()
    cache = load_cache()
    
    # Extract topics from query
    topics = extract_topics(query)
    current_hour = datetime.now().hour
    
    # Score candidate topics from multiple signals
    candidates: dict[str, float] = {}
    reasons: list[str] = []
    
    # Signal 1: Co-occurrence associations
    for topic in topics:
        if topic in associations:
            for related, score in associations[topic].items():
                if related not in topics:  # Don't predict what's already mentioned
                    candidates[related] = candidates.get(related, 0) + score * 0.4
            reasons.append(f"associations from {topic}")
    
    # Signal 2: Sequential patterns (what usually follows)
    sequential = patterns.get("sequential", {})
    for topic in topics:
        if topic in sequential:
            for i, next_topic in enumerate(sequential[topic]):
                if next_topic not in topics:
                    candidates[next_topic] = candidates.get(next_topic, 0) + (0.3 - i * 0.05)
            reasons.append(f"sequential pattern after {topic}")
    
    # Signal 3: Temporal patterns (what's usually queried at this hour)
    temporal = patterns.get("temporal", {})
    hour_key = str(current_hour)
    if hour_key in temporal:
        for t in temporal[hour_key]:
            if t not in topics:
                candidates[t] = candidates.get(t, 0) + 0.15
        reasons.append(f"temporal pattern for hour {current_hour}")
    
    # Signal 4: Heat (hot topics more likely to be relevant)
    heat = patterns.get("heat", {})
    for topic, score in heat.items():
        if topic in candidates:
            candidates[topic] *= (1 + score / 200)  # Boost hot topics up to 1.5x
    
    # Sort and take top N
    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])[:max_predictions]
    predicted_topics = [t[0] for t in sorted_candidates]
    
    # Check cache for predicted topics
    cached_results = []
    for topic in predicted_topics:
        if topic in cache:
            entry = cache[topic]
            # Check TTL
            cached_at = entry.get("cached_at", "")
            ttl_hours = entry.get("ttl_hours", 4)
            try:
                ct = datetime.fromisoformat(cached_at)
                age_hours = (datetime.now() - ct).total_seconds() / 3600
                if age_hours < ttl_hours:
                    cached_results.append({
                        "topic": topic,
                        "results": entry.get("results", []),
                        "age_hours": round(age_hours, 1),
                    })
            except (ValueError, TypeError):
                pass
    
    elapsed_ms = (time.monotonic() - start) * 1000
    confidence = min(1.0, sum(s for _, s in sorted_candidates[:3]) / 2) if sorted_candidates else 0.0
    
    return {
        "input_topics": topics,
        "predicted_topics": predicted_topics,
        "scores": {t: round(s, 3) for t, s in sorted_candidates},
        "cached_results": cached_results,
        "confidence": round(confidence, 2),
        "reasoning": "; ".join(reasons) if reasons else "no patterns yet",
        "elapsed_ms": round(elapsed_ms, 2),
    }


def anticipate_for_briefing() -> dict:
    """
    Special mode: predict what the user will ask about in the morning briefing.
    Combines yesterday's hot topics + routine topics + temporal patterns.
    """
    patterns = load_patterns()
    heat = patterns.get("heat", {})
    temporal = patterns.get("temporal", {})
    
    # Morning hours (8-11)
    morning_topics = set()
    for h in range(8, 12):
        if str(h) in temporal:
            morning_topics.update(temporal[str(h)])
    
    # Add hot topics
    hot = [t for t, s in heat.items() if s > 50]
    morning_topics.update(hot)
    
    # Always include routine
    morning_topics.update(["business", "business", "health"])
    
    return {
        "briefing_topics": list(morning_topics),
        "hot_topics": hot,
        "temporal_morning": list(morning_topics),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--briefing":
            result = anticipate_for_briefing()
            print(json.dumps(result, indent=2))
        else:
            query = " ".join(sys.argv[1:])
            result = anticipate(query)
            print(json.dumps(result, indent=2))
    else:
        print("Usage: python3 anticipator.py <query>")
        print("       python3 anticipator.py --briefing")
        print("\nExample:")
        result = anticipate("Family called about the appointment")
        print(json.dumps(result, indent=2))
