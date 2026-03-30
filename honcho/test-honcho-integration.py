#!/usr/bin/env python3
"""Test the Honcho → OpenClaw integration.
Simulates what the hook does: queries Honcho context + dialectic, writes to memory/honcho-context.md
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

HONCHO_BASE = "http://${HONCHO_URL:-localhost:7780}/v3"
WORKSPACE = "${WORKSPACE_NAME:-memory}"
PEER = "user"
WORKSPACE_DIR = os.path.expanduser("~/.openclaw/workspace")

prompt = sys.argv[1] if len(sys.argv) > 1 else "How is the business business performing?"

print("=== Honcho Integration Test ===")
print(f"Prompt: {prompt}\n")

# Extract search terms
words = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in prompt).split()
search_terms = ' '.join(w for w in words if len(w) > 3)[:200]
print(f"Search terms: {search_terms}\n")

lines = [f"# Honcho User Context — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", f"Prompt: {prompt[:200]}\n"]

# 1. Context endpoint
print("--- Test 1: Peer Context ---")
t0 = time.time()
try:
    url = f"{HONCHO_BASE}/workspaces/{WORKSPACE}/peers/{PEER}/context?" + urllib.parse.urlencode({
        "search_query": search_terms,
        "include_most_frequent": "true",
        "max_conclusions": "15"
    })
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    t1 = time.time()
    rep = data.get("representation", "")
    card = data.get("peer_card", "")
    print(f"✓ Context: {len(rep)} chars in {int((t1-t0)*1000)}ms")
    print(rep[:400] + "...\n")
    if rep:
        lines.append(f"\n## Representation (what Honcho knows about the user)\n\n{rep[:3000]}")
    if card:
        lines.append(f"\n## Peer Card\n\n{card}")
except Exception as e:
    print(f"✗ Context failed: {e}\n")

# 2. Dialectic chat
print("--- Test 2: Dialectic Chat ---")
t0 = time.time()
try:
    url = f"{HONCHO_BASE}/workspaces/{WORKSPACE}/peers/{PEER}/chat"
    payload = json.dumps({
        "query": f'Based on what you know about the user, what context is most relevant to this message: "{prompt[:500]}"',
        "reasoning_level": "minimal"
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    t1 = time.time()
    content = data.get("content", "")
    print(f"✓ Dialectic: {len(content)} chars in {int((t1-t0)*1000)}ms")
    print(content[:400] + "...\n")
    if content:
        lines.append(f"\n## Dialectic Context (Honcho's understanding)\n\n{content[:2000]}")
except Exception as e:
    print(f"✗ Dialectic failed: {e}\n")

# 3. Write file
out_path = os.path.join(WORKSPACE_DIR, "memory", "honcho-context.md")
output = "\n".join(lines)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(output)
print(f"--- Written {len(output)} chars to {out_path} ---")
print("\n=== Test Complete ===")
