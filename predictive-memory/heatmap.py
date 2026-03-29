#!/usr/bin/env python3
"""
Memory Heat Map — Tracks which memories/topics are hot vs cold.
Reinforcement on access, exponential decay over time.

Usage:
  python3 heatmap.py                  # Show current heat map
  python3 heatmap.py --reinforce X    # Bump topic X
  python3 heatmap.py --decay          # Apply daily decay (cron)
  python3 heatmap.py --json           # Output as JSON
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.path.expanduser("~/.openclaw/workspace/memory/predictive"))
HEATMAP_FILE = DATA_DIR / "heatmap.json"

DECAY_RATE = 0.90  # 10% decay per day
REINFORCE_AMOUNT = 10.0  # Points per access
MAX_HEAT = 100.0
COLD_THRESHOLD = 10.0  # Below this = cold
WARM_THRESHOLD = 40.0  # Below this = warm, above = hot


def load_heatmap() -> dict:
    if HEATMAP_FILE.exists():
        try:
            with open(HEATMAP_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    # Seed with known topics
    return {
        "topics": {
            "business": {"heat": 70, "last_access": datetime.now().isoformat(), "access_count": 0},
            "business": {"heat": 65, "last_access": datetime.now().isoformat(), "access_count": 0},
            "health": {"heat": 60, "last_access": datetime.now().isoformat(), "access_count": 0},
            "family": {"heat": 55, "last_access": datetime.now().isoformat(), "access_count": 0},
            "tech": {"heat": 50, "last_access": datetime.now().isoformat(), "access_count": 0},
            "dad": {"heat": 40, "last_access": datetime.now().isoformat(), "access_count": 0},
            "crypto": {"heat": 30, "last_access": datetime.now().isoformat(), "access_count": 0},
            "travel": {"heat": 20, "last_access": datetime.now().isoformat(), "access_count": 0},
            "cars": {"heat": 15, "last_access": datetime.now().isoformat(), "access_count": 0},
        },
        "last_decay": datetime.now().isoformat(),
    }


def save_heatmap(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HEATMAP_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(HEATMAP_FILE)


def reinforce(topic: str):
    """Bump a topic's heat on access."""
    data = load_heatmap()
    topics = data.setdefault("topics", {})
    
    if topic not in topics:
        topics[topic] = {"heat": 0, "last_access": "", "access_count": 0}
    
    entry = topics[topic]
    entry["heat"] = min(MAX_HEAT, entry["heat"] + REINFORCE_AMOUNT)
    entry["last_access"] = datetime.now().isoformat()
    entry["access_count"] = entry.get("access_count", 0) + 1
    
    save_heatmap(data)
    return entry["heat"]


def apply_decay():
    """Apply daily decay to all topics. Run from cron."""
    data = load_heatmap()
    topics = data.get("topics", {})
    
    decayed = 0
    evicted = 0
    
    for topic, entry in list(topics.items()):
        old_heat = entry["heat"]
        entry["heat"] = round(entry["heat"] * DECAY_RATE, 1)
        if entry["heat"] < 1.0:
            # Evict frozen topics
            del topics[topic]
            evicted += 1
        elif entry["heat"] != old_heat:
            decayed += 1
    
    data["last_decay"] = datetime.now().isoformat()
    save_heatmap(data)
    print(f"Decay applied: {decayed} topics decayed, {evicted} evicted")
    return data


def get_status(topic: str) -> str:
    """Return 'hot', 'warm', or 'cold'."""
    data = load_heatmap()
    entry = data.get("topics", {}).get(topic)
    if not entry:
        return "unknown"
    heat = entry["heat"]
    if heat >= WARM_THRESHOLD:
        return "hot"
    elif heat >= COLD_THRESHOLD:
        return "warm"
    return "cold"


def display_heatmap():
    """Pretty-print the heat map."""
    data = load_heatmap()
    topics = data.get("topics", {})
    
    sorted_topics = sorted(topics.items(), key=lambda x: -x[1]["heat"])
    
    print("🔥 Memory Heat Map")
    print("=" * 60)
    
    for topic, entry in sorted_topics:
        heat = entry["heat"]
        status = "🔴" if heat >= WARM_THRESHOLD else "🟡" if heat >= COLD_THRESHOLD else "🔵"
        bar_len = int(heat / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        label = "HOT" if heat >= WARM_THRESHOLD else "WARM" if heat >= COLD_THRESHOLD else "COLD"
        count = entry.get("access_count", 0)
        print(f"  {status} {topic:15s} {bar} {heat:5.1f} ({label}, {count} hits)")
    
    print(f"\nLast decay: {data.get('last_decay', 'never')}")


if __name__ == "__main__":
    if "--reinforce" in sys.argv:
        idx = sys.argv.index("--reinforce")
        if idx + 1 < len(sys.argv):
            topic = sys.argv[idx + 1]
            new_heat = reinforce(topic)
            print(f"Reinforced '{topic}': heat = {new_heat}")
        else:
            print("Usage: --reinforce <topic>")
    elif "--decay" in sys.argv:
        apply_decay()
    elif "--json" in sys.argv:
        data = load_heatmap()
        print(json.dumps(data, indent=2))
    else:
        display_heatmap()
