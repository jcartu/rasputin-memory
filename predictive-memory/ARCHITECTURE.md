# Predictive Memory System — Architecture

## Overview
A predictive caching layer that sits between the agent and the existing memory infrastructure (Qdrant Second Brain + openclaw-mem SQLite). Instead of reactive recall, it **anticipates** what information will be needed and pre-loads it into a fast-access cache.

## Components

### 1. Access Tracker (`access_tracker.py`)
Logs every memory query with timestamp, topic, and result count. Builds the raw data for pattern analysis.
- **Storage**: `~/.openclaw/workspace/memory/predictive/access_log.jsonl`
- **Fields**: timestamp, query, topics_extracted, results_count, session_context

### 2. Pattern Analyzer (`pattern_analyzer.py`)
Mines access logs to discover:
- **Temporal patterns**: "business revenue" queries cluster at 9-10am MSK
- **Sequential patterns**: "family" → planning → appointments → supplements (within same session)
- **Frequency decay**: Exponential decay scoring — recent accesses weight more
- **Co-occurrence**: Topics queried together form clusters

### 3. Context Anticipator (`anticipator.py`)
Real-time component called during conversations:
- Input: current topic/entity mentioned
- Output: top 5 predicted next-needed memory chunks
- Uses association maps built by Pattern Analyzer
- Sub-10ms response (pure dict lookup, no ML inference)

### 4. Pre-fetch Daemon (`prefetch.py`)
Cron-driven (runs every 2 hours + special morning run at 8am MSK):
- Checks what day/time it is → predicts likely queries
- Pre-fetches from Qdrant and caches results
- Morning run: heavier — pulls yesterday's topics + routine queries
- **Cache**: `~/.openclaw/workspace/memory/predictive/cache.json`
- **TTL**: 4 hours default, 12 hours for stable facts

### 5. Heat Map (`heatmap.py`)
Tracks memory temperature:
- **Hot**: Accessed 3+ times in 24h — keep in cache permanently
- **Warm**: Accessed in last 48h — cache with normal TTL  
- **Cold**: Not accessed in 7+ days — evict from cache
- Reinforcement: each access bumps temperature
- Decay: temperature drops 10% per day without access

## Data Flow
```
User message → Agent extracts topic
                  ↓
         Context Anticipator (< 10ms)
           ↓              ↓
    Cache HIT         Cache MISS
    (instant)      → Qdrant search → cache result
         ↓
    Return predicted context + search results
         ↓
    Access Tracker logs the query
         ↓
    Pattern Analyzer updates associations (async)
```

## Integration Points

### With memory_engine.py
Add a `predictive_context(topic)` call at the start of `recall()` that:
1. Checks the pre-fetch cache for relevant entries
2. Returns cached results alongside live search results
3. Logs the access for pattern learning

### With Cron
```
0 8 * * * python3 ~/workspace/tools/predictive-memory/prefetch.py --morning
0 */2 * * * python3 ~/workspace/tools/predictive-memory/prefetch.py --refresh
```

### With openclaw-mem
Read session data to understand conversation flow patterns. The `sessions` and `observations` tables provide tool usage patterns.

## Key Design Decisions
1. **No ML models** — Pure statistical patterns + association maps. Fast, debuggable, zero GPU cost.
2. **JSONL for access logs** — Append-only, easy to analyze, no DB overhead.
3. **JSON for cache** — Simple, readable, atomic writes via temp file + rename.
4. **Aggressive caching** — Better to cache too much than miss. Cache is cheap (~few MB).
5. **Graceful degradation** — If cache is empty/stale, falls through to normal Qdrant search.

## Entity Association Map
Pre-built from entity_graph.json + learned from access patterns:
```json
{
  "family": ["planning", "appointments", "supplements", "documents", "citizenship"],
  "business": ["revenue", "deposits", "platform_a", "platform_b", "licensing"],
  "dad": ["lung transplant", "ipf", "toronto general", "medications"],
  "health": ["testosterone", "hgh", "peptides", "mounjaro", "cgm", "whoop"]
}
```
These expand over time as new co-occurrences are observed.
