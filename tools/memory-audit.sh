#!/bin/bash
# Memory System Health Check — outputs JSON for dashboard integration
# Usage: ./memory-audit.sh [--json] [--verbose]

set -euo pipefail

QDRANT_URL="http://localhost:6333"
BRAIN_URL="${MEMORY_API_URL:-http://${MEMORY_API_HOST:-localhost:7777}}"
COLLECTION="second_brain"
JSON_MODE=false
VERBOSE=false

for arg in "$@"; do
    case $arg in
        --json) JSON_MODE=true ;;
        --verbose) VERBOSE=true ;;
    esac
done

# 1. Qdrant health
QDRANT_DATA=$(curl -s "$QDRANT_URL/collections/$COLLECTION")
POINTS=$(echo "$QDRANT_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
INDEXED=$(echo "$QDRANT_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['indexed_vectors_count'])")
SEGMENTS=$(echo "$QDRANT_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['segments_count'])")
STATUS=$(echo "$QDRANT_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['status'])")

# 2. Enrichment backlog (vectors without importance)
NO_IMP=$(curl -s -X POST "$QDRANT_URL/collections/$COLLECTION/points/count" \
    -H 'Content-Type: application/json' \
    -d '{"filter":{"must_not":[{"key":"importance","range":{"gte":-999999,"lte":999999}}]},"exact":true}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['count'])")

# 3. Brain API health
BRAIN_HEALTH=$(curl -s "$BRAIN_URL/health" 2>/dev/null || echo '{"status":"down"}')

# 4. Embedding service
EMBED_OK=$(curl -s http://${OLLAMA_URL:-localhost:11434}/api/embeddings -d '{"model":"nomic-embed-text","prompt":"healthcheck"}' 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if len(d.get('embedding',[])) == 768 else 'error')" 2>/dev/null || echo "down")

# 5. Snapshot count
SNAP_COUNT=$(curl -s "$QDRANT_URL/collections/$COLLECTION/snapshots" \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin)['result']))" 2>/dev/null || echo "0")
SNAP_LATEST=$(curl -s "$QDRANT_URL/collections/$COLLECTION/snapshots" \
    | python3 -c "import sys,json; s=json.load(sys.stdin)['result']; s.sort(key=lambda x:x['creation_time'],reverse=True); print(s[0]['creation_time'] if s else 'none')" 2>/dev/null || echo "none")

# 6. Search quality (quick test)
SEARCH_SCORE=$(curl -s "$BRAIN_URL/search?q=memory+system+health&limit=1" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('results',d.get('hits',[])); print(r[0].get('score',0) if r else 0)" 2>/dev/null || echo "0")

# 7. PM2 services
RERANKER_STATUS=$(pm2 jlist 2>/dev/null | python3 -c "import sys,json; procs=json.load(sys.stdin); print(next((p['pm2_env']['status'] for p in procs if p['name']=='reranker-gpu1'),'missing'))" 2>/dev/null || echo "unknown")
BRAIN_PM2=$(pm2 jlist 2>/dev/null | python3 -c "import sys,json; procs=json.load(sys.stdin); print(next((p['pm2_env']['status'] for p in procs if p['name']=='hybrid-brain'),'missing'))" 2>/dev/null || echo "unknown")

TIMESTAMP=$(date -Iseconds)

if $JSON_MODE; then
    cat <<EOF
{
  "timestamp": "$TIMESTAMP",
  "qdrant": {
    "status": "$STATUS",
    "points": $POINTS,
    "indexed_vectors": $INDEXED,
    "segments": $SEGMENTS,
    "unenriched_vectors": $NO_IMP,
    "enrichment_pct": $(python3 -c "print(round((1 - $NO_IMP/$POINTS) * 100, 1) if $POINTS > 0 else 0)"),
    "snapshots": $SNAP_COUNT,
    "latest_snapshot": "$SNAP_LATEST"
  },
  "services": {
    "brain_api": $(echo "$BRAIN_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print('\"up\"' if d.get('status')=='ok' else '\"down\"')"),
    "embedding": "$EMBED_OK",
    "reranker": "$RERANKER_STATUS",
    "brain_pm2": "$BRAIN_PM2"
  },
  "quality": {
    "search_top_score": $SEARCH_SCORE
  },
  "alerts": [
    $([ "$STATUS" != "green" ] && echo '"qdrant_status_not_green",' || true)
    $([ "$NO_IMP" -gt 1000 ] && echo '"high_unenriched_backlog",' || true)
    $([ "$EMBED_OK" != "ok" ] && echo '"embedding_service_down",' || true)
    $([ "$RERANKER_STATUS" != "online" ] && echo '"reranker_down",' || true)
  ]
}
EOF
else
    echo "═══════════════════════════════════════════════"
    echo "  MEMORY SYSTEM HEALTH CHECK — $TIMESTAMP"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "📊 Qdrant: $STATUS | $POINTS points | $INDEXED indexed | $SEGMENTS segments"
    echo "📈 Enrichment: $((POINTS - NO_IMP))/$POINTS enriched ($(python3 -c "print(round((1 - $NO_IMP/$POINTS) * 100, 1))")%)"
    echo "💾 Snapshots: $SNAP_COUNT (latest: $SNAP_LATEST)"
    echo "🔧 Services: brain=$BRAIN_PM2 embed=$EMBED_OK reranker=$RERANKER_STATUS"
    echo "🔍 Search quality: top score=$SEARCH_SCORE"
    echo ""
    [ "$NO_IMP" -gt 1000 ] && echo "⚠️  ALERT: $NO_IMP vectors missing importance scores!"
    [ "$STATUS" != "green" ] && echo "🔴 ALERT: Qdrant status is $STATUS!"
    [ "$EMBED_OK" != "ok" ] && echo "🔴 ALERT: Embedding service is $EMBED_OK!"
    echo "═══════════════════════════════════════════════"
fi
