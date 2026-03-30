#!/usr/bin/env python3
"""
Memory Consolidator — Unified consolidation pipeline.

Three modes covering different consolidation needs:

  memory    - 5-pass LLM pipeline: extract facts from daily markdown logs,
              verify, dedup, enrich, and diff against MEMORY.md
  sessions  - Parallel worker pipeline: extract facts from session JSONL
              transcripts and commit directly to Qdrant
  migrate   - Qdrant collection migration: re-embed and merge multiple
              collections into a single second_brain collection

Usage:
  python3 tools/consolidator.py memory               # last 7 days of daily logs
  python3 tools/consolidator.py memory --days 30      # last 30 days
  python3 tools/consolidator.py memory --dry-run      # show what would change
  python3 tools/consolidator.py sessions              # parallel session extraction
  python3 tools/consolidator.py migrate               # collection consolidation
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

# --------------------------------------------------------------------------- #
# Shared config                                                                #
# --------------------------------------------------------------------------- #

WORKSPACE = os.environ.get(
    "WORKSPACE_PATH",
    os.environ.get("WORKSPACE_PATH", "./data"),
)
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
MEMORY_MD = os.path.join(WORKSPACE, "MEMORY.md")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_ALT_URL = os.environ.get("EMBED_ALT_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
TARGET_COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")

# LLM config (for memory and sessions modes)
PROXY_URL = os.environ.get("LLM_API_URL", "http://localhost:11436/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b")

SECTIONS = [
    "Identity", "Core Behavior", "Infrastructure", "Commandments",
    "Model Routing", "Config", "Active Crons", "Business Data MCP",
    "Business Intelligence Suite", "Active Monitoring", "Known Bugs",
    "Pending", "Key Files", "Key Pages", "People",
]


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def llm_call(prompt: str, url: str = "", model: str = "",
             max_tokens: int = 2000, temp: float = 0.2) -> str:
    """Single LLM call through an OpenAI-compatible endpoint."""
    resp = requests.post(
        url or PROXY_URL,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        json={
            "model": model or LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    # Support both Anthropic and OpenAI response shapes
    if "content" in data:
        return "".join(b.get("text", "") for b in data.get("content", []))
    choices = data.get("choices", [{}])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return ""


def parse_json_response(text: str) -> list:
    """Parse JSON array from LLM response, handling markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []


def get_embedding_ollama(text: str, prefix: str = "search_document: ") -> Optional[list]:
    """Get embedding from Ollama."""
    try:
        resp = requests.post(EMBED_URL, json={
            "model": EMBED_MODEL,
            "input": f"{prefix}{text[:2000]}",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data:
            return data["embeddings"][0]
        if "embedding" in data:
            return data["embedding"]
    except Exception:
        pass
    # Fallback to /api/embeddings endpoint
    try:
        resp = requests.post(EMBED_ALT_URL, json={
            "model": EMBED_MODEL,
            "prompt": f"{prefix}{text[:2000]}",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding")
    except Exception:
        return None


def fact_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip()[:200].encode()).hexdigest()


# =========================================================================== #
# MODE: memory — 5-pass consolidation from daily logs → MEMORY.md             #
# =========================================================================== #

def load_daily_files(days: int) -> List[Tuple[str, str]]:
    """Load memory/YYYY-MM-DD*.md files from the last N days."""
    files: List[Tuple[str, str]] = []
    seen: set = set()
    today = datetime.datetime.now().date()
    for i in range(days):
        d = today - datetime.timedelta(days=i)
        for pattern in [f"{d.isoformat()}*.md", f"{d.isoformat()}_*.md"]:
            for path in sorted(glob.glob(os.path.join(MEMORY_DIR, pattern))):
                bn = os.path.basename(path)
                if bn in seen:
                    continue
                seen.add(bn)
                try:
                    with open(path) as f:
                        content = f.read()
                    if content.strip():
                        files.append((bn, content))
                except Exception:
                    pass
    return files


def load_memory_md() -> str:
    try:
        with open(MEMORY_MD) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _pass1_extract(name: str, content: str) -> list:
    """Pass 1: Per-file fact extraction."""
    prompt = f"""Extract important FACTS from this session log. Output a JSON array.

Each object: {{"fact": "<one-line factual statement>", "category": "decision"|"infrastructure"|"person"|"preference"|"business"|"bug"|"config"}}

RULES:
- Only FACTS: names, dates, numbers, decisions, infrastructure changes, new tools, preferences, relationships, business outcomes
- Skip: debugging steps, UI iterations, routine commands, things that are temporary/ephemeral
- Be concise — one line per fact
- If nothing important, return []

FILE: {name}
---
{content[:6000]}"""
    try:
        result = llm_call(prompt, max_tokens=1500)
        return parse_json_response(result)
    except Exception as e:
        print(f"  ⚠️ Failed to parse {name}: {e}")
        return []


def _pass2_verify(facts: list, source_files: List[Tuple[str, str]]) -> list:
    """Pass 2: Verify facts against source, remove LOW confidence."""
    if not facts:
        return []
    batch_size = 25
    all_verified: list = []
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start : batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(facts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} facts)...")
        facts_json = json.dumps(batch, indent=1)
        prompt = f"""Verify these facts against the source text. For each:
1. Check accuracy  2. Fix inaccuracies  3. Rate: HIGH/MEDIUM/LOW  4. Drop LOW

Output JSON array: {{"fact": "...", "category": "...", "confidence": "HIGH"|"MEDIUM"|"LOW"}}

FACTS:
{facts_json[:6000]}

SOURCE CONTEXT:
"""
        for name, content in source_files[:4]:
            prompt += f"\n--- {name} ---\n{content[:2000]}\n"
        try:
            result = llm_call(prompt, max_tokens=2500)
            verified = parse_json_response(result)
            high = [f for f in verified if f.get("confidence") != "LOW"]
            dropped = len(verified) - len(high)
            if dropped > 0:
                print(f"    Removed {dropped} LOW confidence, kept {len(high)}")
            all_verified.extend(high)
        except Exception as e:
            print(f"    ⚠️ Batch verification failed: {e}, keeping batch")
            all_verified.extend(batch)
    return all_verified


def _pass3_dedup(all_facts: list) -> list:
    """Pass 3: Deduplicate and merge related facts."""
    if len(all_facts) <= 5:
        return all_facts
    facts_text = json.dumps(all_facts, indent=1)
    prompt = f"""You have {len(all_facts)} extracted facts. Deduplicate and merge related ones.

RULES:
- Remove exact or near-duplicates (keep the more specific version)
- Merge related facts into comprehensive entries where it makes sense
- Keep all unique facts — don't drop things just to be brief
- Output a JSON array of {{"fact": "...", "category": "..."}}
- Preserve dates and numbers exactly

FACTS:
{facts_text[:10000]}"""
    try:
        result = llm_call(prompt, max_tokens=3000)
        return parse_json_response(result)
    except Exception as e:
        print(f"  ⚠️ Dedup failed: {e}, using raw facts")
        return all_facts


def _pass4_enrich(facts: list) -> list:
    """Pass 4: Cross-reference enrichment."""
    if len(facts) <= 3:
        return facts
    today_str = datetime.datetime.now().strftime("%B %Y")
    batch_size = 30
    all_enriched: list = []
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start : batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(facts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} facts)...")
        facts_json = json.dumps(batch, indent=1)
        prompt = f"""Enhance these facts by identifying relationships and adding context.

1. Identify facts that relate to each other
2. Merge related facts into comprehensive entries where it makes sense
3. Add temporal context like "as of {today_str}" where relevant
4. Link cause-and-effect relationships explicitly

Output JSON array: {{"fact": "...", "category": "..."}}

FACTS:
{facts_json[:8000]}"""
        try:
            result = llm_call(prompt, max_tokens=2500)
            enriched = parse_json_response(result)
            if len(enriched) != len(batch):
                print(f"    Merged {len(batch)} → {len(enriched)} in this batch")
            all_enriched.extend(enriched)
        except Exception as e:
            print(f"    ⚠️ Batch enrichment failed: {e}, keeping batch")
            all_enriched.extend(batch)
    return all_enriched


def _pass5_diff(facts: list, current_memory: str) -> list:
    """Pass 5: Diff against MEMORY.md, return only new entries."""
    facts_text = json.dumps(facts, indent=1)
    sections_list = ", ".join(SECTIONS)
    prompt = f"""Compare these extracted facts against the current MEMORY.md. Return ONLY genuinely NEW or STALE-correcting facts.

Output JSON array:
{{"section": "<section name>", "action": "add"|"update", "entry": "<one-line bullet>", "old_text": "<text to replace, only for updates>"}}

AVAILABLE SECTIONS: {sections_list}
If a fact doesn't fit, use "New".

Skip anything already in MEMORY.md (even if worded differently).
Be aggressive about filtering — only high-value additions.

CURRENT MEMORY.MD:
{current_memory[:8000]}

---
EXTRACTED FACTS:
{facts_text[:8000]}"""
    try:
        result = llm_call(prompt, max_tokens=3000)
        return parse_json_response(result)
    except Exception as e:
        print(f"  ⚠️ Diff failed: {e}")
        return []


def apply_entries(entries: list, current_memory: str, dry_run: bool = False) -> Tuple[int, int, int]:
    """Apply entries to MEMORY.md."""
    if not entries:
        return 0, 0, 0
    if not dry_run:
        shutil.copy2(MEMORY_MD, MEMORY_MD + ".bak")
    lines = current_memory.split("\n")
    added, updated, skipped = 0, 0, 0
    for entry in entries:
        action = entry.get("action", "add")
        section = entry.get("section", "New")
        text = entry.get("entry", "").strip()
        if not text or text in current_memory:
            skipped += 1
            continue
        if dry_run:
            label = "ADD" if action == "add" else "UPDATE"
            print(f"  [{label}] → {section}: {text}")
            added += 1 if action == "add" else 0
            updated += 1 if action == "update" else 0
            continue
        if action == "add":
            section_idx = None
            for i, line in enumerate(lines):
                if section.lower() in line.lower() and line.startswith("#"):
                    section_idx = i
                    break
            if section_idx is not None:
                insert_at = len(lines)
                for i in range(section_idx + 1, len(lines)):
                    if lines[i].startswith("## ") and i > section_idx:
                        insert_at = i
                        break
                bullet = f"- {text}" if not text.startswith("-") else text
                lines.insert(insert_at, bullet)
                added += 1
            else:
                if not any("## New" in line for line in lines):
                    lines.append("\n## New")
                lines.append(f"- {text}" if not text.startswith("-") else text)
                added += 1
        elif action == "update":
            old = entry.get("old_text", "")
            if old:
                for i, line in enumerate(lines):
                    if old.strip() in line:
                        lines[i] = line.replace(old.strip(), text)
                        updated += 1
                        break
    if not dry_run:
        with open(MEMORY_MD, "w") as f:
            f.write("\n".join(lines))
    return added, updated, skipped


def run_memory_mode(args):
    """5-pass consolidation from daily markdown logs into MEMORY.md."""
    print(f"📚 Loading daily files from last {args.days} days...")
    daily_files = load_daily_files(args.days)
    print(f"   Found {len(daily_files)} files")
    if not daily_files:
        print("No daily memory files found.")
        return
    current_memory = load_memory_md()
    print(f"   MEMORY.md: {len(current_memory)} chars\n")

    print(f"🔍 Pass 1: Extracting facts from {len(daily_files)} files...")
    all_facts = []
    for i, (name, content) in enumerate(daily_files):
        facts = _pass1_extract(name, content)
        print(f"  [{i+1}/{len(daily_files)}] {name} → {len(facts)} facts")
        all_facts.extend(facts)
    print(f"\n   Total raw facts: {len(all_facts)}")
    if not all_facts:
        print("✅ Nothing new found.")
        return

    print(f"\n🔎 Pass 2: Verifying {len(all_facts)} facts against source...")
    verified = _pass2_verify(all_facts, daily_files)
    print(f"   After verification: {len(verified)} facts")
    if not verified:
        print("✅ No valid facts after verification.")
        return

    print(f"\n🔄 Pass 3: Deduplicating {len(verified)} facts...")
    merged = _pass3_dedup(verified)
    print(f"   After dedup: {len(merged)} facts")

    print(f"\n🔗 Pass 4: Enriching {len(merged)} facts with cross-references...")
    enriched = _pass4_enrich(merged)
    print(f"   After enrichment: {len(enriched)} facts")

    print("\n📋 Pass 5: Diffing against MEMORY.md...")
    entries = _pass5_diff(merged, current_memory)
    print(f"   New entries to apply: {len(entries)}")
    if not entries:
        print("\n✅ MEMORY.md is up to date.")
        return

    # Save report
    today = datetime.datetime.now().date().isoformat()
    report_file = os.path.join(MEMORY_DIR, f"consolidation-{today}.json")
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump({
            "generated": datetime.datetime.now().isoformat(),
            "days": args.days,
            "files_processed": len(daily_files),
            "raw_facts": len(all_facts),
            "after_verification": len(verified),
            "after_dedup": len(merged),
            "after_enrichment": len(enriched),
            "applied": len(entries),
            "entries": entries,
        }, f, indent=2)

    if args.dry_run:
        print("\n🏷️  DRY RUN — would apply:")
    added, updated, skipped = apply_entries(entries, current_memory, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n   Would add {added}, update {updated}, skip {skipped}")
    else:
        print(f"\n✅ MEMORY.md updated: +{added} added, ~{updated} updated, {skipped} skipped")
        print(f"   Backup: {MEMORY_MD}.bak")
    print(f"   Report: {report_file}")


# =========================================================================== #
# MODE: sessions — Parallel fact extraction from session transcripts → Qdrant  #
# =========================================================================== #

# Session worker config — override via env vars or CLI
SESSIONS_DIR = os.environ.get(
    "SESSIONS_DIR",
    os.environ.get("SESSIONS_DIR", "./data/sessions"),
)
CONSOLIDATION_DIR = os.environ.get(
    "CONSOLIDATION_DIR",
    os.path.expanduser("./memory/consolidation"),
)
MAX_CHARS_PER_SESSION = 30000

# Default LLM endpoints for parallel workers — override via CONSOLIDATOR_ENDPOINTS env var
# (JSON array of {"url": ..., "model": ..., "name": ..., "max_tokens": ...})
_DEFAULT_ENDPOINTS = [
    {"url": "http://localhost:11435/v1/chat/completions", "model": "qwen3.5:122b",
     "name": "122B-gpu0", "max_tokens": 8192},
    {"url": "http://localhost:11437/v1/chat/completions", "model": "qwen3.5:122b",
     "name": "122B-gpu2", "max_tokens": 8192},
]

_EXTRACTION_PROMPT = """You are extracting SPECIFIC, FACTUAL knowledge from a conversation between a user and their AI assistant.

Output ONLY JSON lines. One fact per line. No markdown, no commentary, no wrapping.

Categories: DECISION, TASK_DONE, PREFERENCE, PERSON, BUSINESS, TECHNICAL, PERSONAL, EVENT, INSIGHT

Format per line:
{"category":"<CAT>","fact":"<specific fact with names/numbers/dates>","date":"<YYYY-MM-DD or null>"}

Rules:
- SPECIFIC: include exact numbers, names, ports, paths, versions, amounts
- NO generic observations ("the user uses AI" = useless)
- NO meta-commentary about the conversation itself
- NO duplicates within your output
- Skip if session is just heartbeats, tool noise, or greetings
- If truly nothing valuable: output single line {"category":"SKIP","fact":"no extractable knowledge","date":null}

CONVERSATION:
"""


def _get_session_endpoints() -> list:
    env_endpoints = os.environ.get("CONSOLIDATOR_ENDPOINTS")
    if env_endpoints:
        try:
            return json.loads(env_endpoints)
        except json.JSONDecodeError:
            pass
    return _DEFAULT_ENDPOINTS


def _extract_text(content_raw) -> str:
    if isinstance(content_raw, str):
        return content_raw.strip()
    if isinstance(content_raw, list):
        parts = []
        for part in content_raw:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text", "")
                if t:
                    parts.append(t)
        return " ".join(parts).strip()
    return ""


def _read_session(path: str, max_chars: int = MAX_CHARS_PER_SESSION) -> Optional[str]:
    """Read a session JSONL file, extracting user/assistant messages."""
    messages = []
    total_chars = 0
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("type") != "message":
                    continue
                msg = entry.get("message", {})
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                text = _extract_text(msg.get("content", ""))
                if len(text) < 30:
                    continue
                if any(skip in text for skip in ("HEARTBEAT_OK", "NO_REPLY", "✅ New session started")):
                    continue
                if text.startswith("{") and '"type"' in text[:50]:
                    continue
                if len(text) > 2000:
                    text = text[:2000] + "…"
                entry_text = f"[{role.upper()}]: {text}"
                if total_chars + len(entry_text) > max_chars:
                    break
                messages.append(entry_text)
                total_chars += len(entry_text)
    except Exception:
        return None
    if len(messages) < 3:
        return None
    return "\n".join(messages)


def _commit_to_qdrant(fact_text: str, category: str) -> bool:
    """Commit a fact directly to Qdrant (bypasses A-MAC — facts are already LLM-curated)."""
    embedding = get_embedding_ollama(fact_text)
    if not embedding:
        return False
    # Reject garbage embeddings (near-zero magnitude = model mid-swap)
    mag = math.sqrt(sum(x * x for x in embedding))
    if mag < 0.1:
        return False
    point_id = str(uuid.uuid4())
    try:
        resp = requests.put(f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points", json={
            "points": [{
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "text": f"[{category}] {fact_text}",
                    "source": "consolidator",
                    "category": category,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "fact",
                },
            }],
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def run_sessions_mode(args):
    """Parallel extraction from session JSONL transcripts → Qdrant."""
    out_dir = CONSOLIDATION_DIR
    os.makedirs(out_dir, exist_ok=True)

    progress_file = os.path.join(out_dir, "progress.json")
    output_file = os.path.join(out_dir, "extracted.jsonl")
    log_file = os.path.join(out_dir, "consolidator.log")
    summary_file = os.path.join(out_dir, "CONSOLIDATION-REPORT.md")

    endpoints = _get_session_endpoints()
    max_workers = min(args.workers, len(endpoints)) if args.workers else len(endpoints)

    # Thread-safe state
    log_lock = threading.Lock()
    progress_lock = threading.Lock()
    output_lock = threading.Lock()
    hash_lock = threading.Lock()
    stats_lock = threading.Lock()

    seen_hashes: set = set()
    stats = {"sessions": 0, "facts": 0, "committed": 0, "skipped": 0, "errors": 0}
    processed_set: set = set()
    start_time = time.time()

    def log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with log_lock:
            with open(log_file, "a") as f:
                f.write(line + "\n")
            print(line, flush=True)

    # Load existing progress
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            prog = json.load(f)
        processed_set = set(prog.get("processed", []))
        stats.update(prog.get("stats", {}))

    # Load existing hashes
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen_hashes.add(d.get("hash", fact_hash(d.get("fact", ""))))
                except Exception:
                    pass

    open(log_file, "w").close()
    log("=" * 60)
    log("🧠 MEMORY CONSOLIDATOR — sessions mode")
    log(f"   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"   Workers: {max_workers}, Endpoints: {len(endpoints)}")
    log("=" * 60)

    all_files = sorted(
        glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")),
        key=os.path.getsize,
        reverse=True,
    )
    remaining = [f for f in all_files if f not in processed_set]
    log(f"  Total: {len(all_files)}, Done: {len(processed_set)}, Remaining: {len(remaining)}")

    batch_idx = [0]
    idx_lock = threading.Lock()
    save_counter = [0]

    def save_progress():
        with progress_lock:
            tmp = progress_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"processed": list(processed_set), "stats": dict(stats)}, f)
            os.replace(tmp, progress_file)

    def process_one(fpath, endpoint, wid):
        fname = os.path.basename(fpath)
        # Smaller models get smaller context windows
        char_limit = 8000 if "35B" in endpoint.get("name", "") else MAX_CHARS_PER_SESSION
        text = _read_session(fpath, max_chars=char_limit)
        if not text:
            with progress_lock:
                processed_set.add(fpath)
            with stats_lock:
                stats["skipped"] += 1
            return 0

        fsize = os.path.getsize(fpath)
        log(f"  [W{wid}:{endpoint['name']}] {fname[:12]}… ({fsize//1024}KB)")

        prompt = _EXTRACTION_PROMPT + text
        try:
            payload = {
                "model": endpoint["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.15,
                "max_tokens": endpoint.get("max_tokens", 8192),
                "stream": False,
            }
            if "122B" in endpoint.get("name", ""):
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            resp = requests.post(endpoint["url"], json=payload, timeout=300)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            facts = []
            for fline in raw.split("\n"):
                fline = fline.strip()
                if not fline.startswith("{"):
                    continue
                try:
                    fobj = json.loads(fline)
                    if fobj.get("category") not in (None, "SKIP") and fobj.get("fact") and len(fobj["fact"]) > 15:
                        facts.append(fobj)
                except Exception:
                    continue
        except requests.exceptions.Timeout:
            log(f"  [W{wid}] ⏰ TIMEOUT")
            facts = []
        except Exception as e:
            log(f"  [W{wid}] ⚠️ {str(e)[:120]}")
            facts = []

        new_count = 0
        for fobj in facts:
            ftext = fobj["fact"]
            h = fact_hash(ftext)
            with hash_lock:
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            record = {
                "ts": datetime.datetime.now().isoformat(),
                "category": fobj.get("category", "UNKNOWN"),
                "fact": ftext,
                "date": fobj.get("date"),
                "source": "consolidator",
                "hash": h,
                "session": fname,
            }
            with output_lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            try:
                if _commit_to_qdrant(ftext, record["category"]):
                    with stats_lock:
                        stats["committed"] += 1
                        if stats["committed"] % 100 == 0:
                            log(f"  📊 Committed {stats['committed']} facts so far...")
                            time.sleep(5)
                time.sleep(0.5)
            except Exception:
                pass
            new_count += 1
            with stats_lock:
                stats["facts"] += 1

        with progress_lock:
            processed_set.add(fpath)
        with stats_lock:
            stats["sessions"] = len(processed_set)
        if new_count > 0:
            log(f"  [W{wid}] ✅ +{new_count} facts (total: {stats['facts']})")
        return new_count

    def worker(wid):
        endpoint = endpoints[wid % len(endpoints)]
        while True:
            with idx_lock:
                i = batch_idx[0]
                batch_idx[0] += 1
            if i >= len(remaining):
                break
            try:
                process_one(remaining[i], endpoint, wid)
            except Exception as e:
                log(f"  [W{wid}] ❌ {str(e)[:100]}")
                with stats_lock:
                    stats["errors"] += 1
            with idx_lock:
                save_counter[0] += 1
                if save_counter[0] % 10 == 0:
                    save_progress()

    log(f"\n🚀 Launching {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(max_workers)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                log(f"  ❌ Worker died: {e}")

    save_progress()
    elapsed = time.time() - start_time

    # Write summary report
    cats: Dict[str, int] = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    c = r.get("category", "?")
                    cats[c] = cats.get(c, 0) + 1
                except Exception:
                    pass

    report = f"""# 🧠 Memory Consolidation Report — sessions mode
*{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Summary
- Sessions processed: {len(processed_set)} / {len(all_files)}
- New facts extracted: {stats['facts']}
- Committed to Qdrant: {stats['committed']}
- Skipped (trivial): {stats['skipped']}
- Errors: {stats['errors']}
- Workers: {max_workers}
- Runtime: {elapsed/3600:.1f}h ({elapsed/60:.0f}min)
- Throughput: {len(processed_set)/max(elapsed/60, 0.1):.1f} sessions/min

## Facts by Category
"""
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        report += f"- **{cat}**: {count}\n"
    with open(summary_file, "w") as f:
        f.write(report)

    log("")
    log("=" * 60)
    log(f"✅ DONE — {stats['facts']} facts, {stats['committed']} committed, {elapsed/60:.0f}min")
    log("=" * 60)


# =========================================================================== #
# MODE: migrate — Qdrant collection consolidation + re-embedding              #
# =========================================================================== #

# Collections to delete (empty/stale)
_EMPTY_COLLECTIONS = [
    "research", "default", "general",
]

# Collections to migrate: (name, expected_count, old_dimension)
# Override via MIGRATE_COLLECTIONS env var (JSON array of [name, count, dim] arrays)
_DEFAULT_MIGRATE = [
    ("gmail_import", 0, 1024),
    ("midjourney", 0, 512),
    ("user_memories", 0, 1536),
    ("jarvis_learnings", 0, 384),
    ("health_data", 0, 1024),
]


class CollectionMigrator:
    """Migrates multiple Qdrant collections into a single target collection with re-embedding."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.start_time = time.time()
        self.total_migrated = 0
        self.total_skipped = 0
        self.collections_deleted = 0

    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {msg}")

    def delete_empty_collections(self, collection_names: list):
        self.log(f"Step 1: Deleting {len(collection_names)} empty collections...")
        for name in collection_names:
            if self.dry_run:
                self.log(f"  [DRY RUN] Would delete {name}")
                continue
            try:
                resp = requests.delete(f"{QDRANT_URL}/collections/{name}")
                if resp.status_code in (200, 404):
                    self.log(f"  ✓ Deleted {name}")
                    self.collections_deleted += 1
                else:
                    self.log(f"  ✗ Failed to delete {name}: {resp.status_code}")
            except Exception as e:
                self.log(f"  ✗ Error deleting {name}: {e}")

    def scroll_collection(self, collection_name: str, limit: int = 100):
        offset = None
        while True:
            payload: Dict[str, Any] = {"limit": limit, "with_payload": True, "with_vector": False}
            if offset:
                payload["offset"] = offset
            resp = requests.post(
                f"{QDRANT_URL}/collections/{collection_name}/points/scroll",
                json=payload,
            )
            if resp.status_code != 200:
                self.log(f"  ✗ Scroll failed: {resp.status_code}")
                break
            data = resp.json()
            points = data.get("result", {}).get("points", [])
            if not points:
                break
            yield points
            offset = data.get("result", {}).get("next_page_offset")
            if not offset:
                break

    @staticmethod
    def extract_text(payload: Dict[str, Any]) -> Optional[str]:
        for field in ("text", "content", "message", "body", "description"):
            if field in payload:
                val = payload[field]
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    def _process_batch(self, texts: List[str], points: List[Dict]):
        upsert_points = []
        for text, point in zip(texts, points):
            embedding = get_embedding_ollama(text)
            if not embedding:
                continue
            upsert_points.append({
                "id": point["id"],
                "vector": embedding,
                "payload": point["payload"],
            })
        if not upsert_points:
            return
        resp = requests.put(
            f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points",
            json={"points": upsert_points},
        )
        if resp.status_code != 200:
            self.log(f"  ✗ Upsert failed: {resp.status_code} - {resp.text[:200]}")

    def migrate_collection(self, name: str, expected_count: int, old_dim: int):
        self.log(f"\n📦 Migrating {name} ({expected_count:,} vectors, {old_dim}d → 768d)")
        if self.dry_run:
            self.log("  [DRY RUN] Skipping actual migration")
            return
        migrated = 0
        skipped = 0
        batch_texts: List[str] = []
        batch_points: List[Dict] = []
        try:
            for points in self.scroll_collection(name, limit=100):
                for point in points:
                    text = self.extract_text(point.get("payload", {}))
                    if not text:
                        skipped += 1
                        continue
                    batch_texts.append(text)
                    batch_points.append({
                        "id": point["id"],
                        "payload": {**point.get("payload", {}), "source_collection": name},
                    })
                    if len(batch_texts) >= 200:
                        self._process_batch(batch_texts, batch_points)
                        migrated += len(batch_texts)
                        batch_texts, batch_points = [], []
                        if migrated % 10000 == 0:
                            pct = (migrated / max(expected_count, 1)) * 100
                            self.log(f"  Progress: {migrated:,}/{expected_count:,} ({pct:.1f}%)")
            if batch_texts:
                self._process_batch(batch_texts, batch_points)
                migrated += len(batch_texts)
            self.log(f"  ✓ Migrated {migrated:,}, skipped {skipped:,}")
            self.total_migrated += migrated
            self.total_skipped += skipped
            # Delete old collection after successful migration
            requests.delete(f"{QDRANT_URL}/collections/{name}")
            self.log(f"  ✓ Deleted old collection: {name}")
            self.collections_deleted += 1
        except Exception as e:
            self.log(f"  ✗ Migration failed: {e}")

    def enable_quantization(self):
        self.log("\n🔧 Enabling scalar quantization...")
        if self.dry_run:
            return
        try:
            resp = requests.patch(f"{QDRANT_URL}/collections/{TARGET_COLLECTION}", json={
                "quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}},
            })
            self.log("  ✓ Done" if resp.status_code == 200 else f"  ✗ Failed: {resp.status_code}")
        except Exception as e:
            self.log(f"  ✗ Error: {e}")

    def create_indexes(self):
        self.log("\n🔍 Creating payload indexes...")
        if self.dry_run:
            return
        for field_name, field_schema in [("source", "keyword"), ("date", "keyword"),
                                          ("importance", "integer"), ("source_collection", "keyword")]:
            try:
                resp = requests.put(f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/index", json={
                    "field_name": field_name, "field_schema": field_schema,
                })
                self.log(f"  ✓ {field_name} ({field_schema})" if resp.status_code == 200
                          else f"  ⚠ {field_name}: {resp.status_code}")
            except Exception as e:
                self.log(f"  ✗ {field_name}: {e}")

    def tune_hnsw(self):
        self.log("\n⚙️  Tuning HNSW parameters...")
        if self.dry_run:
            return
        try:
            resp = requests.patch(f"{QDRANT_URL}/collections/{TARGET_COLLECTION}", json={
                "hnsw_config": {"m": 16, "ef_construct": 200},
            })
            self.log("  ✓ m=16, ef_construct=200" if resp.status_code == 200
                      else f"  ✗ Failed: {resp.status_code}")
        except Exception as e:
            self.log(f"  ✗ Error: {e}")

    def run(self, collections_to_delete: list, collections_to_migrate: list):
        self.log("🚀 Starting collection migration")
        self.delete_empty_collections(collections_to_delete)
        self.log(f"\n📦 Migrating {len(collections_to_migrate)} collections...")
        for name, count, dim in collections_to_migrate:
            self.migrate_collection(name, count, dim)
        self.enable_quantization()
        self.create_indexes()
        self.tune_hnsw()
        elapsed = time.time() - self.start_time
        self.log("\n" + "=" * 60)
        self.log("✅ MIGRATION COMPLETE")
        self.log(f"   Time: {elapsed/60:.1f}min | Migrated: {self.total_migrated:,} | "
                 f"Skipped: {self.total_skipped:,} | Deleted: {self.collections_deleted}")
        self.log("=" * 60)


def run_migrate_mode(args):
    """Migrate and consolidate Qdrant collections into second_brain."""
    env_migrate = os.environ.get("MIGRATE_COLLECTIONS")
    if env_migrate:
        try:
            migrate_list = [tuple(x) for x in json.loads(env_migrate)]
        except json.JSONDecodeError:
            migrate_list = _DEFAULT_MIGRATE
    else:
        migrate_list = _DEFAULT_MIGRATE

    env_delete = os.environ.get("DELETE_COLLECTIONS")
    if env_delete:
        try:
            delete_list = json.loads(env_delete)
        except json.JSONDecodeError:
            delete_list = _EMPTY_COLLECTIONS
    else:
        delete_list = _EMPTY_COLLECTIONS

    migrator = CollectionMigrator(dry_run=args.dry_run)
    migrator.run(delete_list, migrate_list)


# =========================================================================== #
# CLI entry point                                                              #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Unified memory consolidation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # memory mode
    p_mem = sub.add_parser("memory", help="5-pass consolidation from daily logs → MEMORY.md")
    p_mem.add_argument("--days", type=int, default=7, help="Days of daily files to process")
    p_mem.add_argument("--dry-run", action="store_true", help="Show changes without applying")

    # sessions mode
    p_sess = sub.add_parser("sessions", help="Parallel fact extraction from session transcripts → Qdrant")
    p_sess.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0=auto)")
    p_sess.add_argument("--dry-run", action="store_true", help="Extract but don't commit to Qdrant")

    # migrate mode
    p_mig = sub.add_parser("migrate", help="Consolidate Qdrant collections into second_brain")
    p_mig.add_argument("--dry-run", action="store_true", help="Show what would be migrated")

    args = parser.parse_args()
    if args.mode == "memory":
        run_memory_mode(args)
    elif args.mode == "sessions":
        run_sessions_mode(args)
    elif args.mode == "migrate":
        run_migrate_mode(args)


if __name__ == "__main__":
    main()
