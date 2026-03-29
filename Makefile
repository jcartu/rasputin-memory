.PHONY: setup start stop test health deps docker quickstart demo

# Install Python dependencies
deps:
	pip install -r requirements.txt

# Full setup: deps + env + docker services
setup: deps
	@test -f .env || cp .env.example .env
	docker-compose up -d
	@echo "Waiting for services..."
	@sleep 3
	@echo "Creating Qdrant collection..."
	@curl -sf -X PUT http://localhost:6333/collections/second_brain \
		-H 'Content-Type: application/json' \
		-d '{"vectors":{"size":768,"distance":"Cosine"},"optimizers_config":{"memmap_threshold":20000}}' \
		2>/dev/null || echo "Collection already exists or Qdrant not ready"
	@echo ""
	@echo "Setup complete. Now:"
	@echo "  1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
	@echo "  2. Pull embedding model: ollama pull nomic-embed-text"
	@echo "  3. Start the brain: make start"

# Start the Hybrid Brain API server
start:
	python3 tools/hybrid_brain.py

# Start with multi-tenant support
start-tenant:
	python3 tools/hybrid_brain_v2_tenant.py

# Start docker services only
docker:
	docker-compose up -d

# Stop docker services
stop:
	docker-compose down

# Health check
health:
	@curl -s http://localhost:7777/health | python3 -m json.tool

# Quick test: search + commit
test:
	@echo "=== Health Check ==="
	@curl -sf http://localhost:7777/health | python3 -m json.tool || echo "Brain API not running"
	@echo ""
	@echo "=== Test Search ==="
	@curl -sf "http://localhost:7777/search?q=test&limit=1" | python3 -m json.tool || echo "Search failed"

# One-command quickstart
quickstart:
	bash quickstart.sh

# Run demo only (skip setup)
demo:
	bash quickstart.sh --demo-only
