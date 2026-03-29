#!/usr/bin/env python3
"""
Memory Consolidation — 5-Pass Pipeline
Extracts facts from daily session logs and updates MEMORY.md.

Pass 1: Per-file fact extraction (each file gets full attention)
Pass 2: Verification against source, confidence rating, remove LOW
Pass 3: Dedup & merge all extracted facts
Pass 4: Cross-reference enrichment, add temporal context, link relationships
Pass 5: Diff against MEMORY.md, apply only genuinely new entries

Usage:
  python3 tools/memory_consolidate.py           # last 7 days
  python3 tools/memory_consolidate.py --days 30 # last 30 days
  python3 tools/memory_consolidate.py --dry-run  # show what would change
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import requests
from datetime import datetime, timedelta

WORKSPACE = os.environ.get("WORKSPACE_PATH", os.path.join(os.path.expanduser("~"), ".openclaw", "workspace"))
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
MEMORY_MD = os.path.join(WORKSPACE, "MEMORY.md")
PROXY_URL = os.environ.get("LLM_API_URL", "http://localhost:11436/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b")

SECTIONS = [
    "Identity", "Core Behavior", "Infrastructure", "Commandments",
    "Model Routing", "Config", "Active Crons", "Business Data MCP",
    "Business Intelligence Suite", "Active Monitoring", "Known Bugs",
    "Pending", "Key Files", "Key Pages", "People"
]


def llm_call(prompt: str, max_tokens: int = 2000, temp: float = 0.2) -> str:
    """Single LLM call through llm-proxy."""
    resp = requests.post(
        PROXY_URL,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": temp},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []))


def parse_json_response(text: str) -> list:
    """Parse JSON array from LLM response, handling markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```\w*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
    # Find the JSON array in case there's preamble
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []


def load_daily_files(days: int) -> list[tuple[str, str]]:
    """Load memory/YYYY-MM-DD*.md files from the last N days."""
    files = []
    seen = set()
    today = datetime.now().date()
    for i in range(days):
        d = today - timedelta(days=i)
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


# ── Pass 1: Per-file extraction ──
def extract_from_file(name: str, content: str) -> list[dict]:
    """Extract facts from a single daily file."""
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
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ Failed to parse {name}: {e}")
        return []


# ── Pass 2: Verification ──
def verify_facts(facts: list[dict], source_files: list[tuple[str, str]]) -> list[dict]:
    """Verify facts against source text snippets, rate confidence, remove LOW confidence."""
    if not facts:
        return []

    # Create a lookup of filename → content snippet
    file_lookup = {name: content[:3000] for name, content in source_files}

    # Process in batches of 25 to avoid timeout
    batch_size = 25
    all_verified = []
    
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(facts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} facts)...")
        
        facts_json = json.dumps(batch, indent=1)
        prompt = f"""Verify these facts against the source text. For each fact:
1. Check if it's accurate against the source (based on filename hints)
2. Fix any inaccuracies
3. Rate confidence: HIGH (directly stated), MEDIUM (clearly implied), LOW (uncertain)
4. Remove LOW confidence facts

Output a JSON array of objects: {{"fact": "...", "category": "...", "confidence": "HIGH"|"MEDIUM"|"LOW"}}

Keep only HIGH and MEDIUM confidence facts in your output.

FACTS TO VERIFY:
{facts_json[:6000]}

SOURCE CONTEXT (first 2000 chars of each file):
"""
        for name, content in source_files[:4]:
            prompt += f"\n--- {name} ---\n{content[:2000]}\n"

        try:
            result = llm_call(prompt, max_tokens=2500)
            verified = parse_json_response(result)
            # Filter out LOW confidence
            high_confidence = [f for f in verified if f.get("confidence") != "LOW"]
            filtered_count = len(verified) - len(high_confidence)
            if filtered_count > 0:
                print(f"    Removed {filtered_count} LOW confidence, kept {len(high_confidence)}")
            all_verified.extend(high_confidence)
        except (json.JSONDecodeError, Exception) as e:
            print(f"    ⚠️ Batch verification failed: {e}, keeping batch")
            all_verified.extend(batch)

    return all_verified


# ── Pass 3: Dedup & merge ──
def dedup_and_merge(all_facts: list[dict]) -> list[dict]:
    """Deduplicate and merge related facts."""
    if len(all_facts) <= 5:
        return all_facts

    facts_text = json.dumps(all_facts, indent=1)
    prompt = f"""You have {len(all_facts)} extracted facts. Deduplicate and merge related ones.

RULES:
- Remove exact or near-duplicates (keep the more specific version)
- Merge related facts into single comprehensive entries where it makes sense
- Keep all unique facts — don't drop things just to be brief
- Output a JSON array of {{"fact": "...", "category": "..."}} — same format as input
- Preserve dates and numbers exactly

FACTS:
{facts_text[:10000]}"""

    try:
        result = llm_call(prompt, max_tokens=3000)
        return parse_json_response(result)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ Dedup failed: {e}, using raw facts")
        return all_facts


# ── Pass 4: Cross-reference enrichment ──
def enrich_with_crossrefs(facts: list[dict]) -> list[dict]:
    """Identify related facts, merge into comprehensive entries, add temporal context."""
    if len(facts) <= 3:
        return facts

    today = datetime.now().strftime("%B %Y")
    
    # Process in batches of 30 to avoid timeout
    batch_size = 30
    all_enriched = []
    
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(facts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} facts)...")
        
        facts_json = json.dumps(batch, indent=1)
        prompt = f"""Enhance these facts by identifying relationships and adding context.

TASKS:
1. Identify facts that relate to each other (same topic, person, project, timeframe)
2. Merge related facts into comprehensive entries where it makes sense
3. Add temporal context like "as of {today}" where relevant
4. Link cause-and-effect relationships explicitly

Output a JSON array with the same format: {{"fact": "...", "category": "..."}}

RULES:
- Don't drop important details when merging — combine them
- Add phrases like "as of {today}", "leading to...", "resulting in...", "consequently..."
- Keep the fact concise but comprehensive

FACTS TO ENRICH:
{facts_json[:8000]}"""

        try:
            result = llm_call(prompt, max_tokens=2500)
            enriched = parse_json_response(result)
            original_count = len(batch)
            new_count = len(enriched)
            if new_count != original_count:
                print(f"    Merged {original_count} → {new_count} in this batch")
            all_enriched.extend(enriched)
        except (json.JSONDecodeError, Exception) as e:
            print(f"    ⚠️ Batch enrichment failed: {e}, keeping batch")
            all_enriched.extend(batch)
    
    return all_enriched


# ── Pass 5: Diff against MEMORY.md and format ──
def diff_and_format(facts: list[dict], current_memory: str) -> list[dict]:
    """Compare facts against MEMORY.md, return only new entries with section placement."""
    facts_text = json.dumps(facts, indent=1)
    sections_list = ", ".join(SECTIONS)

    prompt = f"""Compare these extracted facts against the current MEMORY.md. Return ONLY facts that are genuinely NEW or correct STALE information.

Output a JSON array:
{{"section": "<section name>", "action": "add"|"update", "entry": "<one-line bullet>", "old_text": "<text to replace, only for updates>"}}

AVAILABLE SECTIONS: {sections_list}
If a fact doesn't fit, use "New".

Skip anything already captured in MEMORY.md (even if worded differently).
Skip ephemeral/low-value facts.
Be aggressive about filtering — only high-value additions.

CURRENT MEMORY.MD:
{current_memory[:8000]}

---
EXTRACTED FACTS:
{facts_text[:8000]}"""

    try:
        result = llm_call(prompt, max_tokens=3000)
        return parse_json_response(result)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ Diff failed: {e}")
        return []


def apply_entries(entries: list[dict], current_memory: str, dry_run: bool = False) -> tuple[int, int, int]:
    """Apply entries to MEMORY.md. Returns (added, updated, skipped)."""
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
        if not text:
            skipped += 1
            continue

        # Skip if already present
        if text in current_memory:
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
                if not any("## New" in l for l in lines):
                    lines.append("\n## New")
                bullet = f"- {text}" if not text.startswith("-") else text
                lines.append(bullet)
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


def main():
    parser = argparse.ArgumentParser(description="Multi-pass memory consolidation")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    args = parser.parse_args()

    print(f"📚 Loading daily files from last {args.days} days...")
    daily_files = load_daily_files(args.days)
    print(f"   Found {len(daily_files)} files")

    if not daily_files:
        print("No daily memory files found.")
        sys.exit(0)

    current_memory = load_memory_md()
    print(f"   MEMORY.md: {len(current_memory)} chars\n")

    # ── Pass 1: Per-file extraction ──
    print(f"🔍 Pass 1: Extracting facts from {len(daily_files)} files...")
    all_facts = []
    for i, (name, content) in enumerate(daily_files):
        facts = extract_from_file(name, content)
        if facts:
            print(f"  [{i+1}/{len(daily_files)}] {name} → {len(facts)} facts")
            all_facts.extend(facts)
        else:
            print(f"  [{i+1}/{len(daily_files)}] {name} → 0 facts")

    print(f"\n   Total raw facts: {len(all_facts)}")
    if not all_facts:
        print("✅ Nothing new found.")
        sys.exit(0)

    # ── Pass 2: Verification ──
    print(f"\n🔎 Pass 2: Verifying {len(all_facts)} facts against source...")
    verified = verify_facts(all_facts, daily_files)
    print(f"   After verification: {len(verified)} facts")

    if not verified:
        print("✅ No valid facts after verification.")
        sys.exit(0)

    # ── Pass 3: Dedup & merge ──
    print(f"\n🔄 Pass 3: Deduplicating {len(verified)} facts...")
    merged = dedup_and_merge(verified)
    print(f"   After dedup: {len(merged)} facts")

    # ── Pass 4: Cross-reference enrichment ──
    print(f"\n🔗 Pass 4: Enriching {len(merged)} facts with cross-references...")
    enriched = enrich_with_crossrefs(merged)
    print(f"   After enrichment: {len(enriched)} facts")

    # ── Pass 5: Diff against MEMORY.md ──
    print(f"\n📋 Pass 5: Diffing against MEMORY.md...")
    entries = diff_and_format(merged, current_memory)
    print(f"   New entries to apply: {len(entries)}")

    if not entries:
        print("\n✅ MEMORY.md is up to date.")
        sys.exit(0)

    # Save report
    today = datetime.now().date().isoformat()
    report_file = os.path.join(MEMORY_DIR, f"consolidation-{today}.json")
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "days": args.days,
            "files_processed": len(daily_files),
            "raw_facts": len(all_facts),
            "after_verification": len(verified),
            "after_dedup": len(merged),
            "after_enrichment": len(enriched),
            "applied": len(entries),
            "entries": entries
        }, f, indent=2)

    # Apply
    if args.dry_run:
        print("\n🏷️  DRY RUN — would apply:")
    
    added, updated, skipped = apply_entries(entries, current_memory, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n   Would add {added}, update {updated}, skip {skipped}")
    else:
        print(f"\n✅ MEMORY.md updated: +{added} added, ~{updated} updated, {skipped} skipped")
        print(f"   Backup: {MEMORY_MD}.bak")
    print(f"   Report: {report_file}")


if __name__ == "__main__":
    main()
