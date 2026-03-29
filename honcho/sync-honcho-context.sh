#!/bin/bash
# Sync Honcho peer context to hot-context for OpenClaw session injection
# Fetches the user's peer profile from Honcho and writes a clean markdown file

set -euo pipefail

HONCHO_URL="http://${HONCHO_URL:-localhost:7780}"
WORKSPACE="${WORKSPACE_NAME:-memory}"
PEER="user"
OUTPUT_DIR="$HOME/.openclaw/workspace/memory/hot-context"
OUTPUT_FILE="$OUTPUT_DIR/honcho-profile.md"

mkdir -p "$OUTPUT_DIR"

# Fetch peer context
CONTEXT=$(curl -sf "$HONCHO_URL/v3/workspaces/$WORKSPACE/peers/$PEER/context" 2>/dev/null)
if [ -z "$CONTEXT" ]; then
  echo "ERROR: Failed to fetch Honcho context" >&2
  exit 1
fi

# Extract representation and format
REPR=$(echo "$CONTEXT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
repr_text = d.get('representation', '')
# Take last 50 observations to avoid bloat
lines = repr_text.strip().split('\n')
header_lines = []
obs_lines = []
for line in lines:
    if line.startswith('[2'):
        obs_lines.append(line)
    else:
        header_lines.append(line)
# Keep header + last 50 observations
output_lines = header_lines + obs_lines[-50:]
print('\n'.join(output_lines))
")

# Write formatted output
cat > "$OUTPUT_FILE" << 'HEADER'
# Honcho Profile
> Auto-synced from Honcho peer context. Updated every 30 min.
> Source: 240K+ derived conclusions from conversation history.

HEADER

echo "$REPR" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "_Last synced: $(date -Iseconds)_" >> "$OUTPUT_FILE"

echo "OK: Wrote $(wc -l < "$OUTPUT_FILE") lines to $OUTPUT_FILE"
