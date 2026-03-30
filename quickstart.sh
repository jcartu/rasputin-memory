#!/bin/bash
# RASPUTIN Memory System — Quick Start
# Gets you from zero to working memory in 15 minutes

set -e

echo "🧠 RASPUTIN Memory System — Quick Start"
echo "========================================="
echo ""

# ─── 1. Check prerequisites ─────────────────────────────────────────────────

echo "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || { echo "❌ Docker required. Install: https://docs.docker.com/get-docker/"; exit 1; }
echo "  ✅ Docker"

# Check for docker compose (v2) or docker-compose (v1)
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "❌ Docker Compose required. Install: https://docs.docker.com/compose/install/"
    exit 1
fi
echo "  ✅ Docker Compose ($COMPOSE)"

command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.10+ required."; exit 1; }
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ✅ Python $PYVER"

echo ""

# ─── 2. Environment file ────────────────────────────────────────────────────

if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example"
    echo "   Edit .env if you need non-default ports or models"
else
    echo "📝 .env already exists — keeping your settings"
fi

echo ""

# ─── 3. Start infrastructure ────────────────────────────────────────────────

echo "🐳 Starting Qdrant + FalkorDB + Redis..."
$COMPOSE up -d

# ─── 4. Wait for services ───────────────────────────────────────────────────

echo "⏳ Waiting for services..."
sleep 3

# Wait for Qdrant (max 60s)
TRIES=0
until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do
    TRIES=$((TRIES + 1))
    if [ $TRIES -gt 30 ]; then
        echo "❌ Qdrant failed to start after 60s. Check: $COMPOSE logs qdrant"
        exit 1
    fi
    sleep 2
done
echo "  ✅ Qdrant ready"

# Wait for Redis/FalkorDB (max 30s)
TRIES=0
until $COMPOSE exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; do
    TRIES=$((TRIES + 1))
    if [ $TRIES -gt 15 ]; then
        echo "⚠️  Redis/FalkorDB not responding — graph features may be unavailable"
        break
    fi
    sleep 2
done
if [ $TRIES -le 15 ]; then
    echo "  ✅ Redis ready"
    echo "  ✅ FalkorDB ready (runs on Redis protocol)"
fi

echo ""

# ─── 5. Python dependencies ─────────────────────────────────────────────────

echo "📦 Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt
echo "  ✅ Installed in .venv/"

echo ""

# ─── 6. Ollama + embedding model ────────────────────────────────────────────

if ! command -v ollama >/dev/null 2>&1; then
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  ✅ Ollama already installed"
fi

echo "📥 Pulling nomic-embed-text embedding model..."
echo "   ⚠️  IMPORTANT: Must be v1 (768-dim). v1.5 is INCOMPATIBLE."
ollama pull nomic-embed-text

echo ""

# ─── 7. Create Qdrant collection ────────────────────────────────────────────

echo "🗄️ Setting up Qdrant collection..."
python3 -c "
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
client = QdrantClient('localhost', port=6333)
collections = [c.name for c in client.get_collections().collections]
if 'second_brain' not in collections:
    client.create_collection('second_brain', vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    print('  ✅ Collection \"second_brain\" created (768-dim, cosine)')
else:
    print('  ✅ Collection \"second_brain\" already exists')
"

echo ""

# ─── 8. Create log directory ────────────────────────────────────────────────

mkdir -p logs memory/hot-context

# ─── 9. Done ────────────────────────────────────────────────────────────────

echo "========================================="
echo "🎉 Setup complete!"
echo ""
echo "Start the memory server:"
echo "  source .venv/bin/activate"
echo "  python3 tools/hybrid_brain.py"
echo "  → API at http://localhost:7777"
echo ""
echo "Test it:"
echo "  curl -X POST http://localhost:7777/commit \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"Hello, this is my first memory\", \"source\": \"test\"}'"
echo ""
echo "  curl 'http://localhost:7777/search?q=first+memory&limit=5'"
echo ""
echo "📖 Full guide: docs/GETTING_STARTED.md"
echo "🔧 Configuration: docs/CONFIGURATION.md"
echo "⏰ Cron setup: docs/CRON_JOBS.md"
