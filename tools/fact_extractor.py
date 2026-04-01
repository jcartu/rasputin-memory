#!/usr/bin/env python3
"""
Auto Fact Extraction — Mines session transcripts for personal knowledge about the user.
Runs every 4 hours via cron. Uses llm-proxy with Anthropic format.
Implements 3-pass pipeline: Extract → Verify → Filter

Extracts structured facts and stores them in Qdrant + memory/facts.jsonl

Usage:
  python3 fact_extractor.py                  # Process last 4 hours
  python3 fact_extractor.py --all            # Process ALL sessions (first run)
  python3 fact_extractor.py --hours 24       # Process last 24 hours
"""

import json
import importlib
import os
import sys
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path

try:
    _locking = importlib.import_module("pipeline.locking")
except ModuleNotFoundError:
    _locking = importlib.import_module("tools.pipeline.locking")
acquire_pipeline_lock = _locking.acquire_lock

WORKSPACE = Path(os.environ.get("WORKSPACE_PATH", Path.home() / ".openclaw" / "workspace"))
SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", Path.home() / ".openclaw/agents/main/sessions"))
FACTS_FILE = WORKSPACE / "memory" / "facts.jsonl"
STATE_FILE = WORKSPACE / "memory" / "fact_extractor_state.json"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")

# LLM proxy endpoint (OpenAI-compatible or Anthropic-compatible)
LLM_PROXY_URL = os.environ.get("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:14b")


def load_state():
    """Load extractor state (last run time, processed lines, fact hashes)"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_run": None, "processed_lines": {}, "fact_hashes": []}


def save_state(state):
    """Save extractor state"""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_existing_facts():
    """Load all existing facts from facts.jsonl"""
    facts = []
    if FACTS_FILE.exists():
        with open(FACTS_FILE) as f:
            for line in f:
                try:
                    facts.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return facts


def llm_call(prompt, max_tokens=2000, temp=0.1):
    """Call llm-proxy with Anthropic message format"""
    resp = requests.post(
        LLM_PROXY_URL,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []))


def extract_user_messages(hours=4, process_all=False):
    """Extract user messages from session transcripts"""
    cutoff = None if process_all else (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    messages = []
    for jf in sorted(SESSIONS_DIR.glob("*.jsonl")):
        if jf.name == "sessions.json":
            continue
        try:
            with open(jf) as f:
                for i, line in enumerate(f):
                    try:
                        d = json.loads(line.strip())
                        if d.get("type") != "message":
                            continue

                        ts = d.get("timestamp", "")
                        if cutoff and ts and ts < cutoff:
                            continue

                        msg = d.get("message", {})
                        role = msg.get("role", "")

                        # We want user messages AND assistant messages (facts come from both)
                        if role not in ("user", "assistant"):
                            continue

                        content = msg.get("content", "")
                        if isinstance(content, list):
                            text_parts = [
                                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                            ]
                            content = "\n".join(text_parts)

                        if not isinstance(content, str) or len(content) < 30:
                            continue

                        # Skip system noise, cron outputs, tool results
                        if content.startswith("[System") or content.startswith("[cron:"):
                            continue
                        if "Exec completed" in content[:50] or "Exec failed" in content[:50]:
                            continue

                        messages.append(
                            {
                                "role": role,
                                "text": content[:2000],  # Truncate long messages
                                "ts": ts,
                                "file": jf.name,
                                "line": i,
                            }
                        )
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"  Error reading {jf.name}: {e}")

    return messages


def chunk_messages(messages, chunk_size=20):
    """Group messages into conversation chunks for analysis"""
    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i : i + chunk_size]
        text = "\n".join([f"[{m['role']}] {m['text']}" for m in chunk])
        chunks.append(
            {
                "text": text[:8000],  # Keep under model context
                "ts_start": chunk[0]["ts"],
                "ts_end": chunk[-1]["ts"],
                "count": len(chunk),
            }
        )
    return chunks


def pass1_extract_facts(chunk_text):
    """
    PASS 1: Extract facts with STRICT specificity requirements.
    Only extract facts containing NAMES, DATES, NUMBERS, or SPECIFIC DECISIONS.
    """
    prompt = f"""You are analyzing a conversation to extract PERSONAL FACTS about the user.

⚠️ STRICT RULE - ONLY EXTRACT SPECIFIC FACTS:
- Must contain at least one of: NAMES, DATES, NUMBERS, or SPECIFIC DECISIONS
- If a fact is too vague to be useful to a stranger who doesn't know the user, DON'T include it

✅ GOOD EXAMPLES (include these):
- "The team's quarterly revenue reached $2.3M in Q1 2026"
- "Alice started a new role as VP Engineering at Acme Corp on March 3"
- "The company signed a 3-year contract with Globex for $500K/year"
- "Bob relocated from New York to London in January 2026"
- "The product launch date was moved from April to June 2026"
- "The user's preferred stack is Python + FastAPI + Qdrant + FalkorDB"
- "The team hired 3 new engineers in February, bringing headcount to 12"

❌ BAD EXAMPLES (DO NOT include these):
- "The user has family members" (too vague)
- "The user uses a workspace directory" (obvious/generic)
- "The user is interested in health" (vague, no specifics)
- "The user likes music" (no detail)
- "The user works in business" (meaningless without specifics)
- "The user has pets" (unless you include specific names)

Categories to use:
- Family members: names, relationships, specific details about family
- Health information: specific medications, dosages, conditions, doctors, dates
- Life events: specific dates, places, milestones with concrete details
- Business details: company names, revenue numbers, decisions, partnerships
- Personal preferences: specific brands, types, reasons (not just "likes X")
- Locations: specific places, addresses, travel plans with dates
- Relationships: named friends, colleagues, partners with context
- Hobbies/interests: specific activities, equipment, commitments

Output ONLY a JSON array of objects with "category" and "fact" fields.
If no SPECIFIC facts found (only vague ones), output: []

Conversation to analyze:
{chunk_text}

JSON (empty array if nothing specific):"""

    try:
        response = llm_call(prompt, max_tokens=1500, temp=0.1)

        # Parse JSON from response
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        # Try direct parsing first
        try:
            facts = json.loads(text)
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                facts = json.loads(text[start:end])
                if isinstance(facts, list):
                    return facts
            except Exception:
                pass

        return []
    except Exception as e:
        print(f"  Pass 1 error: {e}")
        return []


def pass2_verify_facts(facts, chunk_text):
    """
    PASS 2: Verify extracted facts against source text.
    Mark each as CONFIRMED, INFERRED, or HALLUCINATED.
    Remove HALLUCINATED facts.
    """
    if not facts:
        return []

    facts_json = json.dumps(facts, indent=2)

    prompt = f"""You are verifying facts extracted from a conversation. For each fact, determine if it is:
- CONFIRMED: Directly stated in the source text
- INFERRED: Reasonably implied but not directly stated
- HALLUCINATED: Not supported by the source text or contradicted by it

SOURCE TEXT:
{chunk_text}

EXTRACTED FACTS:
{facts_json}

Respond with JSON in this format:
[
  {{"fact": "original fact text", "status": "CONFIRMED/INFERRED/HALLUCINATED", "reason": "brief explanation"}},
  ...
]

Only return the JSON array, nothing else."""

    try:
        response = llm_call(prompt, max_tokens=2000, temp=0.1)

        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        verified = json.loads(text)

        # Keep only CONFIRMED and INFERRED facts
        confirmed_facts = []
        for v in verified:
            status = v.get("status", "").upper()
            if status in ("CONFIRMED", "INFERRED"):
                # Find original fact and keep it
                original = next((f for f in facts if f.get("fact") == v.get("fact")), None)
                if original:
                    confirmed_facts.append(original)
            elif status == "HALLUCINATED":
                print(f"    ⚠️ Hallucinated (removed): {v.get('fact', '')[:80]}")

        return confirmed_facts

    except Exception as e:
        print(f"  Pass 2 error: {e}")
        # If verification fails, keep all facts from pass 1
        return facts


def pass3_filter_existing(new_facts, existing_facts):
    """
    PASS 3: Filter new facts against existing facts.
    Remove: already known (even if worded differently), too vague, temporary/ephemeral.
    """
    if not new_facts:
        return []

    new_facts_json = json.dumps([f.get("fact", "") for f in new_facts], indent=2)

    existing_facts_list = [f.get("fact", "") for f in existing_facts]
    existing_json = json.dumps(existing_facts_list[:100], indent=2)  # Limit to avoid overflow

    prompt = f"""You are filtering newly extracted facts against an existing knowledge base.

REMOVE facts that are:
1. Already known (even if worded differently) - check against existing facts below
2. Too vague to be actionable (e.g., "The user is interested in X" without specifics)
3. Temporary/ephemeral (e.g., "The user will do X tomorrow", one-off commands, transient states)

KEEP only facts that are:
- Specific and permanent/long-term
- Add new information not already captured
- Contain names, dates, numbers, or specific decisions

NEW FACTS TO EVALUATE:
{new_facts_json}

EXISTING FACTS (already in database):
{existing_json}

Respond with JSON array containing ONLY the facts to KEEP (not remove).
Return empty array [] if nothing new to keep.

Example response format:
[{{"fact": "The user increased their medication dosage from 5mg to 7.5mg on March 3"}}, ...]

Just return the JSON array:"""

    try:
        response = llm_call(prompt, max_tokens=1500, temp=0.1)

        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        filtered = json.loads(text)

        # Convert back to fact objects
        kept_facts = []
        kept_texts = set(filtered) if isinstance(filtered, list) else set()

        for fact in new_facts:
            fact_text = fact.get("fact", "")
            if fact_text in kept_texts:
                kept_facts.append(fact)

        removed_count = len(new_facts) - len(kept_facts)
        if removed_count > 0:
            print(f"    🔄 Filtered out {removed_count} duplicate/vague/temporary facts")

        return kept_facts

    except Exception as e:
        print(f"  Pass 3 error: {e}")
        # If filtering fails, keep all new facts
        return new_facts


def dedup_fact(fact_text, existing_hashes):
    """Check if we already have this fact"""
    h = hashlib.md5(fact_text.lower().strip().encode()).hexdigest()
    return h in existing_hashes, h


def store_fact(fact, state):
    """Store fact to JSONL file and Qdrant"""
    fact_text = fact.get("fact", "")
    category = fact.get("category", "unknown")

    is_dup, fact_hash = dedup_fact(fact_text, set(state.get("fact_hashes", [])))
    if is_dup:
        return False

    # Append to JSONL
    entry = {"ts": datetime.now().isoformat(), "category": category, "fact": fact_text, "hash": fact_hash}
    with open(FACTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Also commit to Second Brain API for A-MAC scoring + graph extraction
    try:
        brain_resp = requests.post(
            "http://localhost:7777/commit",
            json={"text": f"[{category}] {fact_text}", "source": "fact-extractor"},
            timeout=30,
        )
        if brain_resp.status_code == 200:
            brain_data = brain_resp.json()
            if brain_data.get("ok"):
                print(
                    f"    → Second Brain: accepted (score={brain_data.get('amac', {}).get('scores', {}).get('composite', '?')})"
                )
    except Exception:
        pass

    state.setdefault("fact_hashes", []).append(fact_hash)
    return True


def purge_garbage_facts():
    """
    Purge vague, obvious, and redundant facts from facts.jsonl.
    Returns count of facts purged.
    """
    if not FACTS_FILE.exists():
        return 0

    # Back up first
    backup_path = FACTS_FILE.with_suffix(".jsonl.bak")
    import shutil

    shutil.copy2(FACTS_FILE, backup_path)
    print(f"  Backed up facts to {backup_path.name}")

    all_facts = []
    with open(FACTS_FILE) as f:
        for line in f:
            try:
                all_facts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

    len(all_facts)

    # Vague/obvious patterns to remove
    vague_patterns = [
        "user has family",
        "user uses a workspace",
        "user is interested in health",
        "user uses gpu",
        "user lives in",
        "user works",
        "user likes",
        "user is",
        "user has pets",  # Unless specific names
        "user wants",
        "user can",
        "user knows",
        "user is involved",
    ]

    kept_facts = []
    purged_count = 0

    for fact in all_facts:
        fact_text = fact.get("fact", "").lower()

        # Check for vague patterns
        is_vague = False
        for pattern in vague_patterns:
            if pattern in fact_text and len(fact.get("fact", "")) < 60:
                # Short facts matching vague pattern are likely garbage
                is_vague = True
                break

        # Keep facts with specific names, numbers, dates
        has_specifics = any(
            [
                any(c.isupper() for c in fact.get("fact", "")),  # Has capital letters (names)
                any(c.isdigit() for c in fact.get("fact", "")),  # Has numbers
                "." in fact.get("fact", ""),  # Has decimals (dosages, prices)
            ]
        )

        if not is_vague or has_specifics:
            kept_facts.append(fact)
        else:
            purged_count += 1
            print(f"    🗑️ Removed: {fact.get('fact', '')[:70]}")

    # Write cleaned version
    with open(FACTS_FILE, "w") as f:
        for fact in kept_facts:
            f.write(json.dumps(fact) + "\n")

    print(f"  Purged {purged_count} garbage facts, kept {len(kept_facts)}")
    return purged_count


def main():
    try:
        _lock_fd = acquire_pipeline_lock("fact_extractor")
    except OSError:
        print("[INFO] Another fact_extractor instance is running. Exiting.")
        sys.exit(0)
    process_all = "--all" in sys.argv
    hours = 4
    for i, arg in enumerate(sys.argv):
        if arg == "--hours" and i + 1 < len(sys.argv):
            hours = int(sys.argv[i + 1])

    state = load_state()

    print(f"🧠 Fact Extractor — {'ALL sessions' if process_all else f'last {hours}h'}")
    print(f"  LLM proxy: {LLM_MODEL} via {LLM_PROXY_URL}")
    print(f"  Existing facts: {len(state.get('fact_hashes', []))}")

    # Extract messages
    messages = extract_user_messages(hours=hours, process_all=process_all)
    print(f"  Messages found: {len(messages)}")

    if not messages:
        print("  Nothing to process")
        state["last_run"] = datetime.now().isoformat()
        save_state(state)
        return

    # Chunk and process with 3-pass pipeline
    chunks = chunk_messages(messages, chunk_size=15)
    print(f"  Chunks to analyze: {len(chunks)}")

    total_new = 0
    total_dup = 0
    total_verified = 0
    total_filtered = 0

    # Load existing facts for pass 3
    existing_facts = load_existing_facts()
    print(f"  Existing facts for dedup: {len(existing_facts)}")

    for i, chunk in enumerate(chunks):
        print(f"\n  [{i + 1}/{len(chunks)}] Processing {chunk['count']} messages...")

        # PASS 1: Extract with strict specificity requirements
        pass1_facts = pass1_extract_facts(chunk["text"])
        print(f"    Pass 1 extracted: {len(pass1_facts)} facts")

        if not pass1_facts:
            continue

        # PASS 2: Verify against source text
        pass2_facts = pass2_verify_facts(pass1_facts, chunk["text"])
        total_verified += len(pass1_facts) - len(pass2_facts)
        print(f"    Pass 2 verified: {len(pass2_facts)} (removed {total_verified} hallucinated)")

        if not pass2_facts:
            continue

        # PASS 3: Filter against existing facts
        pass3_facts = pass3_filter_existing(pass2_facts, existing_facts)
        total_filtered += len(pass2_facts) - len(pass3_facts)
        print(f"    Pass 3 filtered: {len(pass3_facts)} (removed {total_filtered} duplicates/vague)")

        # Store remaining facts
        for fact in pass3_facts:
            if not fact.get("fact"):
                continue
            stored = store_fact(fact, state)
            if stored:
                total_new += 1
                existing_facts.append(fact)  # Update local cache
                print(f"      ✅ [{fact.get('category', '?')}] {fact['fact'][:80]}")
            else:
                total_dup += 1

    state["last_run"] = datetime.now().isoformat()
    save_state(state)

    print("\n📊 Results:")
    print(f"  Pass 2 removed (hallucinated): {total_verified}")
    print(f"  Pass 3 removed (duplicates/vague): {total_filtered}")
    print(f"  New facts stored: {total_new}")
    print(f"  Duplicates skipped: {total_dup}")
    print(f"  Total facts in store: {len(state.get('fact_hashes', []))}")


if __name__ == "__main__":
    main()
