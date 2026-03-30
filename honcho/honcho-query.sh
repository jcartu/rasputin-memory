#!/bin/bash
# Query Honcho conclusions by semantic search
# Usage: bash tools/honcho-query.sh "music preferences" [top_k]

set -euo pipefail

QUERY="${1:?Usage: honcho-query.sh \"query string\" [top_k]}"
TOP_K="${2:-10}"
HONCHO_URL="http://${HONCHO_URL:-localhost:7780}"
WORKSPACE="${WORKSPACE_NAME:-memory}"

# Query conclusions (observer=user watching user)
RESULT=$(curl -sf -X POST "$HONCHO_URL/v3/workspaces/$WORKSPACE/conclusions/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"$QUERY\",\"top_k\":$TOP_K,\"filters\":{\"observer_id": "${OBSERVER:-user}\"}}" 2>/dev/null)

COUNT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null)

if [ "$COUNT" = "0" ]; then
  # Try without observer filter (agent observing user)
  RESULT=$(curl -sf -X POST "$HONCHO_URL/v3/workspaces/$WORKSPACE/conclusions/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$QUERY\",\"top_k\":$TOP_K}" 2>/dev/null)
  COUNT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null)
fi

echo "=== Honcho Conclusions: \"$QUERY\" ($COUNT results) ==="
echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if not isinstance(data, list):
    print('Error:', json.dumps(data))
    sys.exit(1)
for i, c in enumerate(data, 1):
    content = c.get('content', 'N/A')
    created = c.get('created_at', '')[:10]
    print(f'{i}. [{created}] {content}')
"
