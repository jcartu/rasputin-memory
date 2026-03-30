#!/usr/bin/env python3
"""
memory-autogen.py — Auto-generate the dynamic section of MEMORY.md from Qdrant.
Runs nightly at 3 AM MSK. Updates only the AUTO-GENERATED block — never touches
the permanent manual sections.

The block is delimited by:
  <!-- AUTO-GEN:START -->
  ...content...
  <!-- AUTO-GEN:END -->

If the markers don't exist yet, they're appended to the end of the file.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.parse
import os

WORKSPACE = Path(os.environ.get("WORKSPACE_PATH", "./data"))
MEMORY_FILE = WORKSPACE / 'MEMORY.md'
QDRANT_SEARCH = 'http://localhost:7777/search'
HOT_DIR = WORKSPACE / 'memory' / 'hot-context'

START_MARKER = '<!-- AUTO-GEN:START -->'
END_MARKER = '<!-- AUTO-GEN:END -->'


def search(query: str, limit: int = 15) -> list:
    try:
        url = f"{QDRANT_SEARCH}?q={urllib.parse.quote(query)}&limit={limit}"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        return data.get('results', [])
    except Exception as e:
        print(f"[autogen] Search failed for '{query}': {e}")
        return []


def load_hot_context() -> list[str]:
    """Load fresh hot-context entries (< 24h old)."""
    entries = []
    if not HOT_DIR.exists():
        return entries
    now = datetime.now(timezone.utc).timestamp()
    for f in sorted(HOT_DIR.glob('*.md')):
        try:
            age = now - f.stat().st_mtime
            if age < 86400:  # 24h
                content = f.read_text(encoding='utf-8').strip()
                # Strip the timestamp comment line
                lines = [line for line in content.splitlines() if not line.startswith('<!--')]
                if lines:
                    entries.append(f"**{f.stem}:** {' '.join(lines[:3])[:300]}")
        except Exception:
            pass
    return entries


def build_autogen_block() -> str:
    now = datetime.now().strftime('%Y-%m-%d %H:%M MSK')
    lines = ["<!-- AUTO-GEN:START -->",
             f"*Auto-generated {now} by memory-autogen.py*\n"]

    # 1. Hot context from recent cron outputs
    hot = load_hot_context()
    if hot:
        lines.append("## 🔥 Recent Intel (last 24h)")
        for h in hot[:6]:
            lines.append(f"- {h}")
        lines.append("")

    # 2. Recent decisions / active work from Qdrant
    recent = search("recent decision active task working on user", limit=12)
    if recent:
        lines.append("## 🧭 Recent Activity (from memory)")
        seen = set()
        count = 0
        for r in recent:
            text = (r.get('text') or r.get('payload', {}).get('text', '')).strip()
            if not text or len(text) < 20:
                continue
            snippet = text[:200].replace('\n', ' ')
            key = snippet[:60]
            if key in seen:
                continue
            seen.add(key)
            score = r.get('score', 0)
            if score < 0.4:
                continue
            lines.append(f"- {snippet}")
            count += 1
            if count >= 6:
                break
        lines.append("")

    # 3. Pending / in-progress items
    pending = search("pending todo in progress waiting deliverable", limit=10)
    if pending:
        lines.append("## ⏳ Pending (from memory)")
        seen = set()
        count = 0
        for r in pending:
            text = (r.get('text') or r.get('payload', {}).get('text', '')).strip()
            if not text or len(text) < 20:
                continue
            snippet = text[:180].replace('\n', ' ')
            key = snippet[:60]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {snippet}")
            count += 1
            if count >= 5:
                break
        lines.append("")

    lines.append(END_MARKER)
    return '\n'.join(lines)


def update_memory_file():
    if not MEMORY_FILE.exists():
        print(f"[autogen] MEMORY.md not found at {MEMORY_FILE}")
        sys.exit(1)

    content = MEMORY_FILE.read_text(encoding='utf-8')
    new_block = build_autogen_block()

    if START_MARKER in content and END_MARKER in content:
        # Replace existing block
        pattern = re.compile(
            re.escape(START_MARKER) + r'.*?' + re.escape(END_MARKER),
            re.DOTALL
        )
        new_content = pattern.sub(new_block, content)
        print("[autogen] ✓ Replaced existing AUTO-GEN block")
    else:
        # Append to end
        new_content = content.rstrip() + '\n\n' + new_block + '\n'
        print("[autogen] ✓ Appended new AUTO-GEN block")

    MEMORY_FILE.write_text(new_content, encoding='utf-8')
    print(f"[autogen] ✓ MEMORY.md updated ({len(new_content)} chars)")


if __name__ == '__main__':
    print(f"[autogen] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    update_memory_file()
    print("[autogen] Done.")
