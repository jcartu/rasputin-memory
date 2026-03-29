#!/usr/bin/env python3
"""
Smart Memory Query - Enhanced second brain querying
Features:
- Multi-query decomposition (break complex queries into sub-queries)
- Deduplication (remove duplicate results)
- Temporal weighting (prefer recent memories)
- Relevance ranking with recency bias
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import math
import hashlib

EMBEDDING_ENDPOINT = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
QDRANT_ENDPOINT = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")

# Topic expansions for multi-query decomposition
TOPIC_EXPANSIONS = {
    "rag": ["retrieval", "vector database", "embeddings", "semantic search"],
    "business": ["dashboard", "grafana", "monitoring", "platform", "analytics"],
    "gpu": ["graphics card", "cuda", "vram", "blackwell", "rtx", "nvidia"],
    "email": ["gmail", "inbox", "messages", "sent"],
    "health": ["fitbit", "workout", "fitness", "sleep", "steps", "glucose"],
    "second brain": ["qdrant", "memory", "embeddings", "memories"],
    "dashboard": ["nexus", "frontend", "ui", "interface"],
    "agent": ["sub-agent", "autonomous", "opencode", "opus"],
    "brazil": ["br", "portuguese", "pt-br", "brazilian"],
    "affiliate": ["propellerads", "media buying", "campaigns"],
}

def get_embedding(text):
    """Get embedding from local embedding service"""
    try:
        cmd = ['curl', '-s', EMBEDDING_ENDPOINT, '-H', 'Content-Type: application/json',
               '-d', json.dumps({"inputs": text[:8000]})]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        return data[0] if isinstance(data, list) and len(data) > 0 else data
    except Exception as e:
        print(f"⚠️  Embedding error: {e}", file=sys.stderr)
        return None

def decompose_query(query: str) -> List[str]:
    """Break complex query into simpler sub-queries"""
    base_query = query.lower().strip()
    queries = [query]  # Start with original
    
    # Add expansions for known topics
    for topic, expansions in TOPIC_EXPANSIONS.items():
        if topic in base_query:
            for expansion in expansions:
                if expansion not in base_query:
                    queries.append(f"{query} {expansion}")
    
    # Limit to 4 queries to avoid spam
    return queries[:4]

def search_memories(query: str, limit: int = 10, score_threshold: float = 0.4) -> List[Dict]:
    """Search second brain for relevant memories"""
    try:
        embedding = get_embedding(query)
        if embedding is None:
            return []
        
        search_payload = {
            "vector": embedding,
            "limit": limit,
            "with_payload": True,
            "score_threshold": score_threshold
        }
        
        cmd = ['curl', '-s', f'{QDRANT_ENDPOINT}/collections/{COLLECTION}/points/search',
               '-H', 'Content-Type: application/json', '-d', json.dumps(search_payload)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        return data.get('result', [])
    except Exception as e:
        print(f"⚠️  Search error: {e}", file=sys.stderr)
        return []

def calculate_content_hash(text: str) -> str:
    """Generate hash for deduplication"""
    # Use first 200 chars to identify duplicates
    normalized = text[:200].strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()

def parse_date(date_str: str) -> datetime:
    """Parse date from various formats"""
    if not date_str:
        return datetime(2000, 1, 1)  # Very old default
    
    try:
        # Try ISO format first
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        try:
            # Try YYYY-MM-DD
            return datetime.strptime(date_str[:10], '%Y-%m-%d')
        except:
            return datetime(2000, 1, 1)

def calculate_temporal_score(date_str: str, recency_weight: float = 0.3) -> float:
    """Calculate temporal decay factor"""
    memory_date = parse_date(date_str)
    age_days = (datetime.now() - memory_date).days
    
    # Exponential decay: e^(-decay_rate * age_days)
    decay_rate = 0.002 * recency_weight
    return math.exp(-decay_rate * age_days)

def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicate memories based on content hash"""
    seen_hashes = set()
    unique = []
    
    for result in results:
        text = result.get('payload', {}).get('text', '')
        content_hash = calculate_content_hash(text)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique.append(result)
    
    return unique

def smart_query(
    query: str,
    max_results: int = 10,
    recency_weight: float = 0.3,
    verbose: bool = False
) -> List[Dict]:
    """
    Enhanced query with multi-query decomposition and temporal awareness
    
    Args:
        query: Search query
        max_results: Maximum results to return
        recency_weight: 0.0 = no temporal bias, 1.0 = only recent matters
        verbose: Print debug info
    """
    if verbose:
        print(f"🧠 Smart query: {query}\n", file=sys.stderr)
    
    # 1. Decompose into sub-queries
    queries = decompose_query(query)
    if verbose and len(queries) > 1:
        print(f"📊 Decomposed into {len(queries)} sub-queries:", file=sys.stderr)
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}", file=sys.stderr)
        print(file=sys.stderr)
    
    # 2. Query each sub-query
    all_results = []
    for sub_query in queries:
        results = search_memories(sub_query, limit=max_results)
        all_results.extend(results)
    
    if verbose:
        print(f"📥 Retrieved {len(all_results)} total results\n", file=sys.stderr)
    
    if not all_results:
        return []
    
    # 3. Deduplicate
    unique_results = deduplicate_results(all_results)
    if verbose:
        print(f"✨ {len(unique_results)} unique after deduplication\n", file=sys.stderr)
    
    # 4. Apply temporal weighting and rerank
    now = datetime.now()
    reranked = []
    
    for result in unique_results:
        payload = result.get('payload', {})
        original_score = result.get('score', 0)
        
        # Get date from payload
        date_str = payload.get('date') or payload.get('timestamp') or ''
        temporal_factor = calculate_temporal_score(date_str, recency_weight)
        
        # Combined score: semantic relevance + temporal factor
        combined_score = (original_score * (1 - recency_weight)) + (temporal_factor * recency_weight)
        
        memory_date = parse_date(date_str)
        age_days = (now - memory_date).days
        
        reranked.append({
            **result,
            'combined_score': combined_score,
            'original_score': original_score,
            'temporal_factor': temporal_factor,
            'age_days': age_days
        })
    
    # 5. Sort by combined score
    reranked.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # 6. Return top N
    return reranked[:max_results]

def format_result(result: Dict, show_scores: bool = False) -> str:
    """Format a single result for display"""
    payload = result.get('payload', {})
    combined_score = result.get('combined_score', 0)
    age_days = result.get('age_days', 0)
    
    # Build header
    parts = []
    if payload.get('date'):
        parts.append(f"[{payload['date'][:10]}]")
    if payload.get('source'):
        parts.append(f"{payload['source']}")
    if payload.get('chat_title'):
        parts.append(f"'{payload['chat_title']}'")
    
    header = " ".join(parts) if parts else "Memory"
    
    # Get text
    text = payload.get('text', '')
    preview = text[:300] + ('...' if len(text) > 300 else '')
    
    # Build output
    output = f"{header}: {preview}"
    
    if show_scores:
        output += f" (score: {combined_score:.3f}, age: {age_days}d)"
    
    return output

def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Smart memory query with multi-query decomposition and temporal weighting'
    )
    parser.add_argument('query', nargs='+', help='Search query')
    parser.add_argument('--max', type=int, default=10, help='Maximum results (default: 10)')
    parser.add_argument('--recency', type=float, default=0.3, 
                       help='Recency weight 0.0-1.0 (default: 0.3)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show debug info')
    parser.add_argument('--scores', '-s', action='store_true',
                       help='Show relevance scores and age')
    
    args = parser.parse_args()
    query = " ".join(args.query)
    
    # Execute smart query
    results = smart_query(
        query,
        max_results=args.max,
        recency_weight=args.recency,
        verbose=args.verbose
    )
    
    if not results:
        print("No relevant memories found.")
        return
    
    # Print results
    if args.verbose:
        print("=" * 80, file=sys.stderr)
        print("TOP RESULTS:", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
    
    print(f"## Relevant Memories ({len(results)} results)\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {format_result(result, show_scores=args.scores)}\n")

if __name__ == "__main__":
    main()
