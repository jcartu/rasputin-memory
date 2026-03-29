#!/usr/bin/env python3
"""
Integration Bridge — Connects the predictive memory system to memory_engine.py.

Drop-in function that can be called from memory_engine.py's recall():
  from predictive_memory.integration import predictive_recall
  extra_context = predictive_recall("family medical appointment")

Also provides a standalone CLI for testing:
  python3 integration.py "what supplements are recommended"
"""

import json
import os
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from access_tracker import log_access, extract_topics
from anticipator import anticipate, load_cache
from heatmap import reinforce


def predictive_recall(query: str, session_id: str = "") -> dict:
    """
    Main integration point. Call this before/alongside Qdrant search.
    
    Returns:
        {
            "predicted_context": [...],  # Pre-fetched results for predicted topics
            "predicted_topics": [...],   # What we think will be asked next
            "cache_hits": int,
            "elapsed_ms": float,
        }
    """
    start = time.monotonic()
    
    # 1. Log the access
    topics = extract_topics(query)
    log_access(query, source="integration", session_id=session_id)
    
    # 2. Reinforce accessed topics in heat map
    for topic in topics:
        reinforce(topic)
    
    # 3. Get predictions
    prediction = anticipate(query)
    
    # 4. Collect cached results for both current and predicted topics
    cache = load_cache()
    context_chunks = []
    cache_hits = 0
    
    # Current topics from cache
    for topic in topics:
        if topic in cache:
            entry = cache[topic]
            for r in entry.get("results", [])[:3]:
                context_chunks.append({
                    "text": r.get("text", ""),
                    "score": r.get("score", 0),
                    "source": f"predictive_cache:{topic}",
                })
                cache_hits += 1
    
    # Predicted topics from cache (lower priority)
    for item in prediction.get("cached_results", []):
        for r in item.get("results", [])[:2]:
            context_chunks.append({
                "text": r.get("text", ""),
                "score": r.get("score", 0) * 0.8,  # Slight discount for predictions
                "source": f"predicted:{item['topic']}",
            })
            cache_hits += 1
    
    elapsed_ms = (time.monotonic() - start) * 1000
    
    return {
        "predicted_context": context_chunks[:10],
        "predicted_topics": prediction.get("predicted_topics", []),
        "scores": prediction.get("scores", {}),
        "confidence": prediction.get("confidence", 0),
        "cache_hits": cache_hits,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def format_for_llm(result: dict) -> str:
    """Format predictive results as context for LLM consumption."""
    chunks = result.get("predicted_context", [])
    if not chunks:
        return ""
    
    lines = ["[Predictive Memory Context]"]
    for chunk in chunks:
        source = chunk.get("source", "")
        text = chunk.get("text", "").strip()
        if text:
            lines.append(f"• ({source}) {text}")
    
    predicted = result.get("predicted_topics", [])
    if predicted:
        lines.append(f"\n[Anticipated topics: {', '.join(predicted)}]")
    
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 integration.py <query>")
        print("\nExample:")
        sys.argv = [sys.argv[0], "family mentioned the appointment"]
    
    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")
    
    result = predictive_recall(query)
    print(f"Elapsed: {result['elapsed_ms']}ms")
    print(f"Cache hits: {result['cache_hits']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Predicted topics: {result['predicted_topics']}")
    print(f"\n--- LLM Context ---")
    print(format_for_llm(result))
