#!/bin/bash
# Test the Honcho → OpenClaw integration
# Simulates what the hook does: queries Honcho context + dialectic and writes to memory/honcho-context.md

set -e

HONCHO_BASE="http://${HONCHO_URL:-localhost:7780}/v3"
WORKSPACE="${WORKSPACE_NAME:-memory}"
PEER="user"
WORKSPACE_DIR="${HOME}/.openclaw/workspace"
TEST_PROMPT="${1:-What are the user's preferences for managing his AI systems?}"

echo "=== Honcho Integration Test ==="
echo "Prompt: $TEST_PROMPT"
echo ""

# Extract search terms
SEARCH_TERMS=$(echo "$TEST_PROMPT" | tr -d '[:punct:]' | tr ' ' '\n' | awk 'length>3' | head -12 | tr '\n' ' ')
echo "Search terms: $SEARCH_TERMS"
echo ""

# 1. Test context endpoint
echo "--- Test 1: Peer Context (representation + peer card) ---"
CONTEXT_START=$(date +%s%N)
CONTEXT_RESP=$(curl -s "${HONCHO_BASE}/workspaces/${WORKSPACE}/peers/${PEER}/context?search_query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$SEARCH_TERMS'))")&include_most_frequent=true&max_conclusions=15" 2>/dev/null)
CONTEXT_END=$(date +%s%N)
CONTEXT_MS=$(( (CONTEXT_END - CONTEXT_START) / 1000000 ))
REP_LEN=$(echo "$CONTEXT_RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('representation','')))" 2>/dev/null)
echo "✓ Context response: ${REP_LEN} chars in ${CONTEXT_MS}ms"
echo "$CONTEXT_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin).get('representation',''); print(r[:500]+'...')" 2>/dev/null
echo ""

# 2. Test dialectic chat endpoint
echo "--- Test 2: Dialectic Chat ---"
CHAT_START=$(date +%s%N)
CHAT_RESP=$(curl -s "${HONCHO_BASE}/workspaces/${WORKSPACE}/peers/${PEER}/chat" \
  -X POST -H "Content-Type: application/json" \
  -d "{\"query\": \"Based on what you know about the user, what context is most relevant to this message: \\\"${TEST_PROMPT}\\\"\", \"reasoning_level\": \"minimal\"}" 2>/dev/null)
CHAT_END=$(date +%s%N)
CHAT_MS=$(( (CHAT_END - CHAT_START) / 1000000 ))
CHAT_LEN=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('content','')))" 2>/dev/null)
echo "✓ Dialectic response: ${CHAT_LEN} chars in ${CHAT_MS}ms"
echo "$CHAT_RESP" | python3 -c "import sys,json; c=json.load(sys.stdin).get('content',''); print(c[:500]+'...')" 2>/dev/null
echo ""

# 3. Write to honcho-context.md (simulating hook behavior)
echo "--- Test 3: Write honcho-context.md ---"
python3 -c "
import json, sys, datetime
context = json.loads('''${CONTEXT_RESP}''') if '''${CONTEXT_RESP}''' else {}
chat = json.loads('''${CHAT_RESP}''') if '''${CHAT_RESP}''' else {}

lines = [f'# Honcho User Context — {datetime.datetime.now().isoformat()}\n']
lines.append(f'Prompt: ${TEST_PROMPT[:200]}\n')

rep = context.get('representation', '')
if rep:
    lines.append(f'\n## Representation (what Honcho knows about the user)\n\n{rep[:3000]}')

card = context.get('peer_card', '')
if card:
    lines.append(f'\n## Peer Card\n\n{card}')

content = chat.get('content', '')
if content:
    lines.append(f'\n## Dialectic Context (Honcho understanding)\n\n{content[:2000]}')

output = '\n'.join(lines)
with open('${WORKSPACE_DIR}/memory/honcho-context.md', 'w') as f:
    f.write(output)
print(f'✓ Written {len(output)} chars to memory/honcho-context.md')
" 2>/dev/null

echo ""
echo "=== Integration Test Complete ==="
echo "Context: ${CONTEXT_MS}ms | Dialectic: ${CHAT_MS}ms | Total: $(( CONTEXT_MS + CHAT_MS ))ms"
echo "File: ${WORKSPACE_DIR}/memory/honcho-context.md"
