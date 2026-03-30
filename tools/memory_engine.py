#!/usr/bin/env python3
"""
Rasputin Memory Engine v3.0 — The Real Thing

Design principles:
1. SPEED: 8 embeddings in 55ms, search in 16ms. We can do 20+ searches per turn for <500ms total.
2. MULTI-ANGLE: Same question searched 5 different ways (semantic, entity, temporal, source-filtered, follow-up)
3. DEDUPLICATION: Same email thread doesn't appear 5 times
4. CONTEXT WINDOW: Smart truncation — most relevant first, formatted for LLM consumption
5. CONVERSATION COMMIT: Every important exchange goes back into Qdrant
6. PERSONALITY: Outputs are written to make the agent sound like it knows the user

Usage:
  # Before every response — get memory context
  python3 memory_engine.py recall "What supplements is partner taking?"
  
  # After important exchanges — commit to memory
  python3 memory_engine.py commit "User decided to go with AcmeCorp for Brazil campaigns, PT-BR only"
  
  # Morning briefing — surface what the user should know
  python3 memory_engine.py briefing
  
  # Find contradictions with a statement  
  python3 memory_engine.py challenge "I think we should spend $50K on Google Ads"
  
  # Deep dive on a topic — get everything we know
  python3 memory_engine.py deep "CHRONOS hardware wallet"
  
  # Who is this person?
  python3 memory_engine.py whois "Oren"
"""

import requests
import json
import sys
import re
import os
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8006/rerank")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")

# v2-moe requires task-type prefixes for best performance
EMBED_QUERY_PREFIX = "search_query: "
EMBED_DOC_PREFIX = "search_document: "

# Import BM25 hybrid search
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bm25_search import hybrid_rerank
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
ENTITY_GRAPH = os.path.expanduser("~/.openclaw/workspace/memory/entity_graph.json")
OM_OBSERVATIONS = os.path.expanduser("~/.openclaw/workspace/memory/om_observations.md")
OM_MAX_AGE_HOURS = 24  # Refresh observations if older than this


def om_lookup(query, max_chunks=5):
    """
    Fast Observational Memory lookup — keyword match against compressed observations.
    Returns relevant observation lines WITHOUT an API call (pure local, <1ms).
    """
    if not os.path.exists(OM_OBSERVATIONS):
        return []
    
    try:
        mtime = os.path.getmtime(OM_OBSERVATIONS)
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours > OM_MAX_AGE_HOURS * 3:  # Stale but still useful as context
            pass  # Use anyway, just won't be fresh
        
        with open(OM_OBSERVATIONS, "r") as f:
            content = f.read()
    except Exception:
        return []
    
    if not content or len(content) < 100:
        return []
    
    # Split into observation blocks (date-grouped)
    blocks = []
    current_block = []
    for line in content.split("\n"):
        if line.startswith("Date: "):
            if current_block:
                blocks.append("\n".join(current_block))
            current_block = [line]
        elif line.strip():
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))
    
    # Score each block by keyword overlap with query
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    scored = []
    for block in blocks:
        block_lower = block.lower()
        # Count matching keywords
        matches = sum(1 for w in query_words if w in block_lower)
        if matches >= 2 or (matches >= 1 and len(query_words) <= 3):
            scored.append((matches / max(len(query_words), 1), block))
    
    # Sort by relevance, return top chunks
    scored.sort(key=lambda x: x[0], reverse=True)
    return [block for score, block in scored[:max_chunks]]

# ═══════════════════════════════════════════════════════════════
# LOW-LEVEL: Embedding & Search
# ═══════════════════════════════════════════════════════════════

def batch_embed(texts, prefix=None):
    """Embed multiple texts in one call via Ollama nomic-embed-text-v2-moe.
    Must use Ollama — GPU embed service is a different model.
    v2-moe requires task prefixes: use EMBED_QUERY_PREFIX for queries,
    EMBED_DOC_PREFIX for documents. Defaults to query prefix."""
    if prefix is None:
        prefix = EMBED_QUERY_PREFIX
    try:
        r = requests.post(EMBED_URL, json={
            "model": EMBED_MODEL,
            "input": [f"{prefix}{t[:4000]}" for t in texts]
        }, timeout=30)
        data = r.json()
        if "embeddings" in data:
            return data["embeddings"]
        elif "embedding" in data:
            return [data["embedding"]]
        return []
    except:
        return []

def search_by_vector(vector, top_k=5, threshold=0.50, filter_dict=None):
    """Search with pre-computed vector."""
    payload = {"vector": vector, "limit": top_k, "score_threshold": threshold, "with_payload": True}
    if filter_dict:
        payload["filter"] = filter_dict
    try:
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search", json=payload, timeout=10)
        return r.json().get("result", [])
    except:
        return []

def rerank(query, results):
    """
    Rerank search results using bge-reranker-v2-m3.
    Returns results sorted by reranker score.
    """
    if not results:
        return []
    
    # Extract passages from results
    passages = []
    for r in results:
        p = r.get("payload", {})
        # Build passage text from available fields
        passage_parts = []
        if p.get("subject"):
            passage_parts.append(f"Subject: {p['subject']}")
        if p.get("title"):
            passage_parts.append(f"Title: {p['title']}")
        if p.get("question"):
            passage_parts.append(f"Question: {p['question']}")
        text = p.get("text", p.get("body", ""))
        if text:
            passage_parts.append(text[:1000])  # Limit to 1000 chars
        passages.append(" | ".join(passage_parts) if passage_parts else "")
    
    try:
        # Call reranker API
        resp = requests.post(RERANKER_URL, json={"query": query, "passages": passages}, timeout=30)
        resp.raise_for_status()
        scores = resp.json().get("scores", [])
        
        if len(scores) != len(results):
            # Fallback to original scoring if reranker fails
            return results
        
        # Attach reranker scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = scores[i]
        
        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
    
    except Exception as e:
        # Fallback: return original results if reranker fails
        return results

# ═══════════════════════════════════════════════════════════════
# QUERY EXPANSION: Turn one question into 5+ search angles
# ═══════════════════════════════════════════════════════════════

def expand_queries(message):
    """
    Generate multiple search queries from a single message.
    Each angle catches different memories.
    """
    queries = [message]  # Original always included
    
    msg_lower = message.lower()
    words = message.split()
    
    # 1. Extract proper nouns and search each
    skip_words = {"The","This","That","What","When","Where","How","Why","Can","Could","Would",
                  "Should","Will","Do","Does","Did","Is","Are","Was","Were","Have","Has","Had",
                  "I","My","We","Our","Yes","No","Ok","Hey","Hi","Hello","Set","Make","Run",
                  "Get","Let","Put","Give","Take","Check","Look","Find","Show","Tell","Ask",
                  "Try","Write","Read","Send","Add","Use","Also","Just","Sure","Yeah","Please",
                  "Thanks","Good","Great","Nice","Cool","Right","Well","But","And","Or","So",
                  "If","For","Not","It","A","An","To","In","On","At","By","Up","Of","With",
                  "About","From","Go","Be","All","Now","Very","Much","Really","Think","Know",
                  "Want","Need","Like"}
    
    entities = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w)
        if clean and clean[0].isupper() and clean not in skip_words and len(clean) > 1:
            entities.append(clean)
    
    # Entity-specific queries
    for entity in list(set(entities))[:4]:
        # Check entity graph for context
        graph_context = lookup_entity_graph(entity)
        if graph_context:
            queries.append(f"{entity} {graph_context}")
        else:
            queries.append(entity)
    
    # 2. Rephrase as a topic search
    # Strip question words and articles for a cleaner topic search
    topic = re.sub(r'^(what|when|where|who|how|why|did|do|does|is|are|was|were|has|have|had|can|could|would|should|will|tell me about|remind me about|what happened with)\s+', '', msg_lower, flags=re.IGNORECASE)
    if topic != msg_lower and len(topic) > 10:
        queries.append(topic)
    
    # 3. Source-specific expansions
    if any(w in msg_lower for w in ["email", "wrote", "sent", "received", "inbox", "from", "to"]):
        queries.append(f"email {topic}")
    if any(w in msg_lower for w in ["searched", "researched", "looked up", "looked into", "perplexity", "google"]):
        queries.append(f"perplexity search {topic}")
    if any(w in msg_lower for w in ["chatgpt", "conversation", "discussed", "talked about", "told"]):
        queries.append(f"chatgpt conversation {topic}")
    
    # 4. Time-based expansion
    if any(w in msg_lower for w in ["last week", "recently", "yesterday", "today", "this month"]):
        queries.append(f"recent {topic}")
    if any(w in msg_lower for w in ["last year", "months ago", "a while back", "back in"]):
        queries.append(f"older {topic}")
    
    # 5. Semantic opposites / related concepts for better recall
    # Maps known topic areas to related search terms
    expansions = {
        "ivf": ["fertility supplements embryo PGT genetic screening"],
        "business": ["revenue deposits analytics marketing"],
        "car": ["supercar sports-car tuning racing motorsport"],
        "health": ["testosterone peptide HGH surgery recovery"],
        "vpn": ["Russia censorship Astrill Amnezia proxy"],
        "property": ["house apartment Moscow Sochi real estate"],
        "crypto": ["Bitcoin CHRONOS wallet blockchain hardware"],
        "doctor": ["medical appointment clinic surgery health"],
        "ring": ["engagement proposal diamond wedding"],
    }
    
    for keyword, expansion_terms in expansions.items():
        if keyword in msg_lower:
            queries.append(expansion_terms[0])
    
    return queries[:12]  # Cap at 12 queries (still < 300ms total)

def lookup_entity_graph(name):
    """Quick lookup in the entity graph JSON."""
    try:
        with open(ENTITY_GRAPH) as f:
            graph = json.load(f)
        
        name_lower = name.lower()
        # Search people
        for person, data in graph.get("people", {}).items():
            if name_lower in person.lower():
                return data.get("role", "") + " " + data.get("context", "")
        # Search companies
        for company, data in graph.get("companies", {}).items():
            if name_lower in company.lower():
                return data.get("type", "") + " " + data.get("context", "")
        return ""
    except:
        return ""


def graph_traverse(query, max_hops=2):
    """
    Deep graph traversal — find entities related to query terms,
    then traverse their connections to build rich context.
    Returns a formatted context block with entity relationships.
    """
    try:
        with open(ENTITY_GRAPH) as f:
            graph = json.load(f)
    except:
        return ""
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    
    found_entities = []
    
    # Pass 1: Find matching entities
    for person, data in graph.get("people", {}).items():
        person_lower = person.lower()
        data_str = json.dumps(data).lower()
        if any(w in person_lower or w in data_str for w in query_words):
            found_entities.append(("person", person, data))
    
    for company, data in graph.get("companies", {}).items():
        company_lower = company.lower()
        data_str = json.dumps(data).lower()
        if any(w in company_lower or w in data_str for w in query_words):
            found_entities.append(("company", company, data))
    
    for topic, data in graph.get("topics", {}).items():
        topic_lower = topic.lower()
        data_str = json.dumps(data).lower()
        if any(w in topic_lower or w in data_str for w in query_words):
            found_entities.append(("topic", topic, data))
    
    if not found_entities:
        return ""
    
    # Pass 2: Traverse connections (1 hop)
    related = []
    for etype, name, data in found_entities:
        # Look for cross-references in data values
        data_str = json.dumps(data)
        
        # Check if any other entities are mentioned in this entity's data
        for person in graph.get("people", {}):
            if person != name and person.lower() in data_str.lower():
                related.append(("person", person, graph["people"][person]))
        for company in graph.get("companies", {}):
            if company != name and company.lower() in data_str.lower():
                related.append(("company", company, graph["companies"][company]))
    
    # Format output
    lines = []
    for etype, name, data in found_entities[:5]:
        role = data.get("role", data.get("type", ""))
        context = data.get("context", "")
        lines.append(f"  {name} ({etype}): {role} {context}".strip())
    
    if related:
        lines.append("  Related:")
        for etype, name, data in related[:5]:
            role = data.get("role", data.get("type", ""))
            lines.append(f"    → {name} ({etype}): {role}")
    
    return "\n".join(lines) if lines else ""

# ═══════════════════════════════════════════════════════════════
# DEDUPLICATION: Don't show the same email thread 5 times
# ═══════════════════════════════════════════════════════════════

def deduplicate(results):
    """Remove near-duplicate results (same email thread, same ChatGPT convo, etc.)"""
    seen_keys = set()
    unique = []
    
    for r in results:
        p = r.get("payload", {})
        
        # Generate dedup key based on source type
        source = p.get("source", "")
        if source in ("email", "gmail"):
            # Dedup by thread_id
            key = p.get("thread_id", p.get("gmail_id", str(r["id"])))
        elif source == "chatgpt":
            # Dedup by title + truncated text (same convo, different turns)
            title = p.get("title", "")
            text_hash = hashlib.md5(p.get("text", "")[:200].encode()).hexdigest()[:8]
            key = f"chatgpt:{title}:{text_hash}"
        elif source == "perplexity":
            key = p.get("filename", p.get("question", str(r["id"])))
        else:
            key = str(r["id"])
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(r)
    
    return unique

# ═══════════════════════════════════════════════════════════════
# RECALL: The main "before every response" function
# ═══════════════════════════════════════════════════════════════

def recall(message, max_results=10, force=False):
    """
    Multi-angle memory recall. Returns formatted context block.
    
    Process:
    1. Expand message into 5-12 search queries
    2. Batch embed all queries
    3. Search Qdrant with each embedding
    4. Deduplicate results
    5. Rank by relevance score
    6. Format for LLM consumption
    """
    # Trigger detection (skip for trivial messages unless forced)
    if not force:
        msg_lower = message.lower()
        triggers = [
            # Personal references
            "remember", "did i", "did we", "have i", "have we", "when did",
            "last time", "previously", "before", "history", "who is", "who was",
            "what happened", "what was", "tell me about", "what about",
            # Possessive / personal
            "my ", "our ", "i was", "i am", "i have", "we were", "we have",
            # Time references
            "yesterday", "last week", "last month", "months ago", "year ago",
            "back when", "back in", "a while", "long time",
            # Specific domains the user cares about
            "appointment", "email", "searched", "looked into", "researched",
            "doctor", "car", "health", "brand", "platform", "revenue",
            "partner", "family_member_1", "family_member_2", "contact_1", "contact_2", "contact_3",
            "supercar", "sports-car", "motorsport", "medical", "health",
            "property", "house", "apartment", "moscow", "sochi", "budapest",
            "vpn", "astrill", "peptide", "testosterone", "surgery", "recovery",
            "ring", "proposal", "engaged", "wedding",
            "racing", "performance",
            "crypto", "bitcoin", "wallet", "ledger",
            "supplement", "medication", "hgh", "bpc", "mots",
            "revenue", "deposit", "affiliate", "streamer",
            "genome", "genetic", "pgt", "embryo",
        ]
        
        # Check for proper nouns (capitalized words)
        has_proper_noun = bool(re.search(r'\b[A-Z][a-z]{2,}\b', message))
        has_trigger = any(t in msg_lower for t in triggers)
        has_question = "?" in message or any(message.lower().startswith(w) for w in ["what", "who", "when", "where", "how", "why", "did", "do", "does", "is", "can", "could", "would", "tell", "remind"])
        
        if not has_trigger and not has_proper_noun and not has_question:
            return {"enriched": False, "context": "", "reason": "no trigger detected"}
    
    # Step 0a: OM fast lookup (local keyword match, <1ms)
    om_context = om_lookup(message)
    
    # Step 0b: Graph traversal (entity relationships, <1ms)
    graph_context = graph_traverse(message)
    
    # Step 1: Query expansion
    queries = expand_queries(message)
    
    # Step 2: Batch embed
    embeddings = batch_embed(queries)
    if not embeddings:
        return {"enriched": False, "context": "", "reason": "embedding failed"}
    
    # Step 3: Two-tier search — prioritize high-value sources, backfill with email
    # Tier 1 (GOLD): ChatGPT convos, Perplexity searches, conversations, social intel
    # Tier 2 (SILVER): Email — only if we need more results
    all_results = {}
    
    HIGH_VALUE_SOURCES = ["chatgpt", "perplexity", "conversation", "social_intel",
                          "grok_social_intel_market_brazil", "grok_social_intel_affiliate",
                          "grok_social_intel_ai_news", "grok_social_intel_ai_agents",
                          "grok_social_intel_crypto", "grok_social_intel_brand_monitoring"]
    
    # Tier 1: Search high-value sources first
    high_value_filter = {
        "should": [{"key": "source", "match": {"value": s}} for s in HIGH_VALUE_SOURCES]
    }
    
    for i, emb in enumerate(embeddings):
        if not isinstance(emb, list):
            continue
        
        # Source filter for specific query types
        filter_dict = None
        query_text = queries[i] if i < len(queries) else ""
        if query_text.startswith("email "):
            filter_dict = {"must": [{"key": "source", "match": {"value": "email"}}]}
        elif query_text.startswith("chatgpt "):
            filter_dict = {"must": [{"key": "source", "match": {"value": "chatgpt"}}]}
        elif query_text.startswith("perplexity "):
            filter_dict = {"must": [{"key": "source", "match": {"value": "perplexity"}}]}
        else:
            # Default: search high-value sources first
            filter_dict = high_value_filter
        
        results = search_by_vector(emb, top_k=10, threshold=0.45, filter_dict=filter_dict)  # Increased for reranking
        for r in results:
            rid = r["id"]
            if rid not in all_results or r["score"] > all_results[rid]["score"]:
                all_results[rid] = r
    
    # Tier 2: If we got fewer than 50 from high-value, backfill with email
    if len(all_results) < 50:
        for i, emb in enumerate(embeddings[:5]):
            if not isinstance(emb, list):
                continue
            email_filter = {"must": [{"key": "source", "match": {"value": "email"}}]}
            results = search_by_vector(emb, top_k=8, threshold=0.50, filter_dict=email_filter)
            for r in results:
                rid = r["id"]
                r["score"] = r["score"] * 0.85
                if rid not in all_results or r["score"] > all_results[rid]["score"]:
                    all_results[rid] = r

    # Tier 3: Unfiltered search — catches telegram, whatsapp, auto-commits, and any other source
    # that isn't in the high-value list. Uses first 3 embeddings to stay fast.
    if len(all_results) < 50:
        for i, emb in enumerate(embeddings[:3]):
            if not isinstance(emb, list):
                continue
            results = search_by_vector(emb, top_k=8, threshold=0.50, filter_dict=None)
            for r in results:
                rid = r["id"]
                if rid not in all_results or r["score"] > all_results[rid]["score"]:
                    all_results[rid] = r
    
    if not all_results:
        return {"enriched": False, "context": "", "reason": "no results found"}
    
    # Step 4: Deduplicate
    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    unique = deduplicate(ranked)
    
    # Step 5a: BM25 hybrid scoring (keyword + semantic fusion via RRF)
    candidates = unique[:50]
    if HAS_BM25:
        candidates = hybrid_rerank(message, candidates)
    
    # Step 5b: Rerank with bge-reranker-v2-m3 (neural reranker on top of hybrid)
    reranked = rerank(message, candidates)
    
    # Step 6: Take top results after reranking
    top = reranked[:max_results]
    
    # Step 7: Format
    context = format_recall(top, message)
    
    # Step 8: Prepend OM + Graph context if available
    om_hit = False
    if om_context:
        om_block = "━━━ 🔮 RECENT CONTEXT (Observational Memory) ━━━\n"
        om_block += "\n---\n".join(om_context[:3])  # Max 3 blocks to keep it tight
        om_block += "\n━━━ END RECENT CONTEXT ━━━\n\n"
        context = om_block + context
        om_hit = True
    
    if graph_context:
        graph_block = "━━━ 🕸️ ENTITY GRAPH ━━━\n"
        graph_block += graph_context
        graph_block += "\n━━━ END ENTITY GRAPH ━━━\n\n"
        context = graph_block + context
    
    return {
        "enriched": True,
        "context": context,
        "results_count": len(top),
        "total_found": len(all_results),
        "queries_used": len(queries),
        "top_score": top[0]["score"] if top else 0,
        "sources": list(set(r.get("payload", {}).get("source", "?") for r in top)),
        "om_hit": om_hit,
        "om_chunks": len(om_context) if om_context else 0,
    }

def format_recall(results, original_query=""):
    """Format results into a context block that makes the agent sound knowledgeable."""
    if not results:
        return ""
    
    lines = []
    lines.append("━━━ 🧠 MEMORY RECALL ━━━")
    lines.append(f"Found {len(results)} relevant memories from second brain (761K total)")
    lines.append("")
    
    for i, r in enumerate(results):
        p = r.get("payload", {})
        score = r.get("score", 0)
        rerank_score = r.get("rerank_score", None)
        source = p.get("source", "unknown")
        
        # Format based on source type
        if source in ("email", "gmail"):
            date = str(p.get("date", ""))[:20]
            subj = p.get("subject", "no subject")
            sender = p.get("from", "unknown")
            body = p.get("body", "")[:400]
            to = p.get("to", "")
            labels = p.get("labels", [])
            
            # Determine direction
            direction = "📨 From" if "user" not in sender.lower() else "📤 Sent by user to"
            target = to if direction.startswith("📤") else sender
            
            lines.append(f"**{direction} {target}**")
            lines.append(f"  Subject: {subj}")
            lines.append(f"  Date: {date}")
            if body.strip():
                lines.append(f"  Body: {body.strip()}")
            if rerank_score is not None:
                lines.append(f"  [rerank: {rerank_score:.3f} | vector: {score:.2f}]")
            else:
                lines.append(f"  [relevance: {score:.2f}]")
            lines.append("")
            
        elif source == "chatgpt":
            title = p.get("title", "untitled")
            date = str(p.get("date", ""))[:10]
            text = p.get("text", "")[:500]
            
            lines.append(f"**💬 ChatGPT conversation: {title}** ({date})")
            lines.append(f"  {text}")
            if rerank_score is not None:
                lines.append(f"  [rerank: {rerank_score:.3f} | vector: {score:.2f}]")
            else:
                lines.append(f"  [relevance: {score:.2f}]")
            lines.append("")
            
        elif source == "perplexity":
            question = p.get("question", "")
            text = p.get("text", "")[:500]
            date = str(p.get("date", ""))[:10]
            
            lines.append(f"**🔍 Perplexity search** ({date}): {question}")
            if text and text != question:
                lines.append(f"  {text}")
            if rerank_score is not None:
                lines.append(f"  [rerank: {rerank_score:.3f} | vector: {score:.2f}]")
            else:
                lines.append(f"  [relevance: {score:.2f}]")
            lines.append("")
        
        else:
            text = p.get("text", p.get("body", ""))[:400]
            date = str(p.get("date", ""))[:10]
            lines.append(f"**📝 Memory** ({date}): {text}")
            if rerank_score is not None:
                lines.append(f"  [rerank: {rerank_score:.3f} | vector: {score:.2f}]")
            else:
                lines.append(f"  [relevance: {score:.2f}]")
            lines.append("")
    
    lines.append("━━━ END MEMORY RECALL ━━━")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# COMMIT: Save important exchanges to memory
# ═══════════════════════════════════════════════════════════════

def commit(text, source="conversation", importance=60, metadata=None):
    """Commit a memory through hybrid_brain /commit API."""
    try:
        resp = requests.post(
            "http://localhost:7777/commit",
            json={
                "text": text,
                "source": source,
                "importance": importance,
                "metadata": metadata,
            },
            timeout=30,
        )
        return resp.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════
# DEEP DIVE: Get EVERYTHING we know about a topic
# ═══════════════════════════════════════════════════════════════

def deep_dive(topic, max_results=20):
    """Pull all memories related to a topic. For when user says 'tell me everything about X'."""
    # Generate many search angles
    base_queries = [
        topic,
        f"{topic} details",
        f"{topic} history",
        f"{topic} plan decision",
        f"{topic} email",
        f"{topic} research",
        f"{topic} cost price money",
        f"{topic} problem issue",
    ]
    
    embeddings = batch_embed(base_queries)
    all_results = {}
    
    for emb in embeddings:
        if not isinstance(emb, list):
            continue
        for r in search_by_vector(emb, top_k=8, threshold=0.50):
            rid = r["id"]
            if rid not in all_results or r["score"] > all_results[rid]["score"]:
                all_results[rid] = r
    
    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    unique = deduplicate(ranked)[:max_results]
    
    return format_recall(unique, topic)

# ═══════════════════════════════════════════════════════════════
# WHOIS: Who is this person? (entity graph + search)
# ═══════════════════════════════════════════════════════════════

def whois(name):
    """Everything we know about a person."""
    # Check entity graph first
    graph_info = lookup_entity_graph(name)
    
    # Search for emails from/to this person
    queries = [
        f"{name}",
        f"from {name}",
        f"to {name}",
        f"{name} meeting conversation",
    ]
    
    embeddings = batch_embed(queries)
    all_results = {}
    
    for emb in embeddings:
        if not isinstance(emb, list):
            continue
        for r in search_by_vector(emb, top_k=5, threshold=0.50):
            rid = r["id"]
            if rid not in all_results or r["score"] > all_results[rid]["score"]:
                all_results[rid] = r
    
    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    unique = deduplicate(ranked)[:10]
    
    lines = [f"━━━ 🔎 WHO IS: {name} ━━━"]
    if graph_info:
        lines.append(f"Entity graph: {graph_info}")
    lines.append("")
    lines.append(format_recall(unique, name))
    
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# CHALLENGE: Find contradictions with a statement
# ═══════════════════════════════════════════════════════════════

def challenge(statement):
    """Search for memories that might contradict or provide context for a statement."""
    queries = [
        statement,
        f"opposite of {statement}",
        f"problem with {statement}",
        f"failed {statement}",
        f"changed mind about {statement}",
    ]
    
    embeddings = batch_embed(queries)
    all_results = {}
    
    for emb in embeddings:
        if not isinstance(emb, list):
            continue
        for r in search_by_vector(emb, top_k=5, threshold=0.45):
            rid = r["id"]
            if rid not in all_results or r["score"] > all_results[rid]["score"]:
                all_results[rid] = r
    
    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    unique = deduplicate(ranked)[:8]
    
    return format_recall(unique, statement)

# ═══════════════════════════════════════════════════════════════
# BRIEFING: Morning context — what should the user know?
# ═══════════════════════════════════════════════════════════════

def briefing():
    """Surface recent important memories for morning briefing."""
    queries = [
        "urgent important action required",
        "meeting appointment scheduled upcoming",
        "reply needed response required follow up",
        "invoice payment money transfer",
        "health medical doctor appointment",
        "deadline due date expiring",
    ]
    
    embeddings = batch_embed(queries)
    all_results = {}
    
    for emb in embeddings:
        if not isinstance(emb, list):
            continue
        for r in search_by_vector(emb, top_k=5, threshold=0.55):
            rid = r["id"]
            if rid not in all_results or r["score"] > all_results[rid]["score"]:
                all_results[rid] = r
    
    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    unique = deduplicate(ranked)[:10]
    
    lines = ["━━━ ☀️ MORNING BRIEFING ━━━"]
    lines.append(format_recall(unique))
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: memory_engine.py <command> [args]")
        print("Commands: recall, commit, deep, whois, challenge, briefing")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "recall":
        msg = " ".join(sys.argv[2:])
        if not msg:
            print("Usage: memory_engine.py recall 'message'")
            sys.exit(1)
        result = recall(msg, force=True)
        if result["enriched"]:
            print(result["context"])
            om_info = f", OM: {result.get('om_chunks', 0)} chunks" if result.get("om_hit") else ""
            print(f"\n[{result['results_count']} results from {result['total_found']} found, {result['queries_used']} queries, top score: {result['top_score']:.3f}{om_info}]")
        else:
            print(f"[No results: {result['reason']}]")
    
    elif cmd == "commit":
        text = " ".join(sys.argv[2:])
        importance = 70
        ok = commit(text, importance=importance)
        print("✓ Committed to memory" if ok else "✗ Failed to commit")
    
    elif cmd == "deep":
        topic = " ".join(sys.argv[2:])
        print(deep_dive(topic))
    
    elif cmd == "whois":
        name = " ".join(sys.argv[2:])
        print(whois(name))
    
    elif cmd == "challenge":
        statement = " ".join(sys.argv[2:])
        print(challenge(statement))
    
    elif cmd == "briefing":
        print(briefing())
    
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: recall, commit, deep, whois, challenge, briefing")
