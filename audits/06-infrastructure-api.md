# Infrastructure & API Audit — rasputin-memory

**Date:** 2026-03-30  
**Auditor:** PhD-level Computer Scientist (Infrastructure/Security Domain)  
**Scope:** HTTP API, Docker setup, deployment, configuration, operational concerns

---

## Executive Summary

The rasputin-memory system is a sophisticated hybrid search engine combining Qdrant (vectors), FalkorDB (graph), BM25 (keyword), and neural reranking. The architecture is well-designed but has **significant operational gaps** in security, error handling, and production readiness.

**Overall Assessment:**
- ✅ **Architecture:** Solid hybrid search pipeline with proper component integration
- ⚠️ **Security:** Multiple critical vulnerabilities (no auth, no input sanitization, runs as root)
- ⚠️ **Observability:** Basic health checks exist but lack metrics, structured logging, tracing
- ⚠️ **Error Handling:** Graceful degradation present but inconsistent timeout/retry patterns
- ⚠️ **Docker:** Functional but not production-optimized (single-stage, runs as root)

**Critical Issues Found:** 3  
**High Severity:** 7  
**Medium Severity:** 12  
**Low Severity:** 8

---

## 1. API Surface Analysis

### Endpoints Summary

| Endpoint | Method | Purpose | Auth | Rate Limit |
|----------|--------|---------|------|------------|
| `/health` | GET | Health check with component status | ❌ No | ❌ No |
| `/stats` | GET | Collection/graph statistics | ❌ No | ❌ No |
| `/search` | GET | Vector search (query params) | ⚠️ Optional token | ❌ No |
| `/search` | POST | Vector search (JSON body) | ⚠️ Optional token | ❌ No |
| `/graph` | GET | Graph traversal search | ❌ No | ❌ No |
| `/commit` | POST | Memory commit with A-MAC gate | ⚠️ Optional token | ❌ No |
| `/proactive` | POST | Proactive memory surfacing | ⚠️ Optional token | ❌ No |
| `/amac/metrics` | GET | A-MAC admission metrics | ❌ No | ❌ No |

### 🔴 CRITICAL Issues

#### 1.1 No Rate Limiting on Any Endpoint
**File:** `tools/hybrid_brain.py` (lines 1-50, entire HTTP handler)  
**Severity:** 🔴 CRITICAL

**Problem:**
```python
class HybridHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # No rate limiting, no request throttling
        result = commit_memory(text, source=source, importance=importance)
```

The API has **zero rate limiting** on all endpoints. A single client can:
- Exhaust embedding service (Ollama) with rapid `/commit` calls
- Flood Qdrant with write operations
- Block neural reranker queue
- Cause denial of service via `/search` with large limits

**Fix:**
```python
from functools import wraps
import time

class RateLimiter:
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.history = []
    
    def allow(self, client_ip: str) -> bool:
        now = time.time()
        self.history = [t for t in self.history if now - t < self.period]
        if len(self.history) >= self.calls:
            return False
        self.history.append(now)
        return True

# In HybridHandler:
RATE_LIMITERS = {
    '/commit': RateLimiter(calls=10, period=60.0),  # 10/min
    '/search': RateLimiter(calls=50, period=60.0),  # 50/min
}

def _check_rate_limit(self, endpoint: str, client_ip: str) -> bool:
    limiter = RATE_LIMITERS.get(endpoint)
    if limiter and not limiter.allow(client_ip):
        self._send_json({"error": "Rate limit exceeded"}, 429)
        return False
    return True
```

**Effort:** 2-3 hours

---

#### 1.2 Optional Authentication — Default Disabled
**File:** `tools/hybrid_brain.py` (lines 835-850)  
**Severity:** 🔴 CRITICAL

**Problem:**
```python
MEMORY_API_TOKEN = os.environ.get("MEMORY_API_TOKEN", "")

def _check_auth(self):
    """Check bearer token if MEMORY_API_TOKEN is set."""
    if not MEMORY_API_TOKEN:
        return True  # ⚠️ Auth disabled by default!
```

**Authentication is opt-in via environment variable.** If `MEMORY_API_TOKEN` is not set (default), **ALL endpoints are completely open**. This includes:
- `/commit` — Anyone can inject memories
- `/search` — Data exfiltration via unlimited queries
- `/amac/metrics` — Internal metrics exposure

**Fix:**
```python
# .env.example
MEMORY_API_TOKEN_REQUIRED=true  # NEW: enforce auth
MEMORY_API_TOKEN=your-secret-token-here

# In hybrid_brain.py
MEMORY_API_TOKEN_REQUIRED = os.environ.get("MEMORY_API_TOKEN_REQUIRED", "true") == "true"
MEMORY_API_TOKEN = os.environ.get("MEMORY_API_TOKEN", "")

def _check_auth(self):
    if MEMORY_API_TOKEN_REQUIRED and not MEMORY_API_TOKEN:
        # Fail-safe: if auth required but no token configured, deny all
        self._send_json({"error": "Server misconfigured: auth required but no token set"}, 500)
        return False
    if not MEMORY_API_TOKEN:
        return True  # Only allow if explicitly disabled
    auth = self.headers.get("Authorization", "")
    if auth == f"Bearer {MEMORY_API_TOKEN}":
        return True
    self._send_json({"error": "Unauthorized"}, 401)
    return False
```

**Effort:** 1 hour

---

#### 1.3 No Input Validation/Sanitization
**File:** `tools/hybrid_brain.py` (lines 870-920, commit handler)  
**Severity:** 🟠 HIGH

**Problem:**
```python
elif parsed.path == "/commit":
    text = data.get("text", "")
    source = data.get("source", "conversation")
    importance = data.get("importance", 60)
    metadata = data.get("metadata", None)
    force = bool(data.get("force", False))

    if not text:
        self._send_json({"error": "Missing text"}, 400)
        return
```

**Missing validations:**
1. **Text length:** No max length check — can submit 10MB texts
2. **Source validation:** Any string accepted (path traversal risk if used in logs)
3. **Importance range:** Can submit `-9999` or `99999`
4. **Metadata injection:** Arbitrary JSON accepted, potentially injected into Qdrant payload
5. **No XSS sanitization:** Text stored as-is, later rendered in dashboards/UIs

**Fix:**
```python
MAX_TEXT_LENGTH = 8000  # chars
MAX_SOURCE_LENGTH = 50
VALID_SOURCES = {"conversation", "email", "telegram", "whatsapp", "perplexity", 
                 "chatgpt", "fact_extractor", "social_intel", "web_page"}
MIN_IMPORTANCE = 0
MAX_IMPORTANCE = 100

def _validate_commit(self, data: dict) -> tuple[bool, str]:
    text = data.get("text", "")
    if not text or len(text) > MAX_TEXT_LENGTH:
        return False, f"Text must be 1-{MAX_TEXT_LENGTH} chars (got {len(text)})"
    
    source = data.get("source", "conversation")
    if len(source) > MAX_SOURCE_LENGTH:
        return False, f"Source too long (max {MAX_SOURCE_LENGTH} chars)"
    
    importance = data.get("importance", 60)
    try:
        importance = int(importance)
        if importance < MIN_IMPORTANCE or importance > MAX_IMPORTANCE:
            return False, f"Importance must be {MIN_IMPORTANCE}-{MAX_IMPORTANCE}"
    except (ValueError, TypeError):
        return False, "Importance must be integer"
    
    # Sanitize metadata
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return False, "Metadata must be object"
    # Strip dangerous keys
    dangerous_keys = {'_id', '__proto__', 'constructor', '$where'}
    for key in dangerous_keys & set(metadata.keys()):
        del metadata[key]
    
    return True, ""
```

**Effort:** 2 hours

---

### 🟠 HIGH Severity Issues

#### 1.4 No Request Size Limits
**File:** `tools/hybrid_brain.py` (lines 860-865)  
**Severity:** 🟠 HIGH

**Problem:**
```python
def do_POST(self):
    content_length = int(self.headers.get("Content-Length", 0))
    body = self.rfile.read(content_length)
```

No `Content-Length` cap. A client can send:
- 100MB JSON body → memory exhaustion
- Slowloris-style tiny chunks → connection starvation

**Fix:**
```python
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB

def do_POST(self):
    content_length = int(self.headers.get("Content-Length", 0))
    if content_length > MAX_BODY_SIZE:
        self._send_json({"error": "Request too large"}, 413)
        return
    body = self.rfile.read(content_length)
```

**Effort:** 30 minutes

---

#### 1.5 No CORS Configuration
**File:** `tools/hybrid_brain.py` (entire file)  
**Severity:** 🟠 HIGH

**Problem:** No CORS headers. If deployed behind a reverse proxy or exposed to browsers:
- Cross-origin attacks possible
- No `Access-Control-Allow-Origin` headers
- No preflight handling

**Fix:**
```python
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",")

def _set_cors_headers(self):
    origin = self.headers.get("Origin", "")
    if origin in CORS_ORIGINS or not CORS_ORIGINS:
        self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    
def do_OPTIONS(self):
    self._set_cors_headers()
    self.send_response(200)
    self.end_headers()
```

**Effort:** 1 hour

---

#### 1.6 No Timeout Configuration on HTTP Server
**File:** `tools/hybrid_brain.py` (lines 950-960)  
**Severity:** 🟠 HIGH

**Problem:**
```python
def serve(port=7777):
    server = ReusableHTTPServer(("127.0.0.1", port), HybridHandler)
    server.serve_forever()
```

No read/write timeouts. Slow clients can:
- Hold connections indefinitely (Slowloris)
- Block worker threads
- Exhaust connection pool

**Fix:**
```python
import socket

class TimeoutHTTPServer(ThreadingHTTPServer):
    request_timeout = 30  # seconds
    allow_reuse_address = True
    
    def finish_request(self, request, client_address):
        try:
            request.settimeout(self.request_timeout)
            super().finish_request(request, client_address)
        except socket.timeout:
            print("[HTTP] Request timeout", flush=True)
        except Exception:
            pass

def serve(port=7777):
    server = TimeoutHTTPServer(("127.0.0.1", port), HybridHandler)
    print(f"[HybridBrain] Serving on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
```

**Effort:** 1 hour

---

#### 1.7 No HTTPS/TLS Support
**File:** Entire deployment — no TLS configuration  
**Severity:** 🟠 HIGH

**Problem:** All communication is plaintext HTTP. If deployed on any network (even internal):
- Credentials/tokens can be intercepted
- Memory data exposed in transit
- No certificate validation

**Fix:** Use reverse proxy (nginx/Caddy) or add TLS:
```python
import ssl

def serve(port=7777):
    server = ReusableHTTPServer(("0.0.0.0", port), HybridHandler)
    
    # TLS wrapper (production: use reverse proxy instead)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile="/path/to/cert.pem", keyfile="/path/to/key.pem")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    
    server.serve_forever()
```

**Effort:** 2 hours (or 30 min for nginx reverse proxy)

---

#### 1.8 No API Versioning
**File:** All endpoints — no version prefix  
**Severity:** 🟡 MEDIUM

**Problem:** Endpoints are `/search`, `/commit`, etc. No versioning means:
- Breaking changes force migration pain
- No deprecation path
- Client compatibility unclear

**Fix:**
```python
# Prefix all routes
elif parsed.path.startswith("/v1/search"):
    # v1 logic
elif parsed.path.startswith("/v2/search"):
    # v2 logic with breaking changes
```

**Effort:** 4 hours (refactor all routes)

---

#### 1.9 No Request Logging/Metrics
**File:** `tools/hybrid_brain.py` line 818  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
def log_message(self, fmt, *args):
    pass  # ⚠️ All logging disabled!
```

**Complete logging suppression** means:
- No request audit trail
- No performance monitoring
- No debugging production issues
- No anomaly detection

**Fix:**
```python
import logging
import time

logger = logging.getLogger("hybrid_brain")

class HybridHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Override to add structured logging
        client_ip = self.client_address[0]
        path = self.path.split("?")[0]
        method = self.command
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        logger.info({
            "timestamp": timestamp,
            "client_ip": client_ip,
            "method": method,
            "path": path,
            "status": self._status_code if hasattr(self, '_status_code') else 200,
        })
    
    def _send_json(self, data, status=200):
        self._status_code = status  # Track for logging
        # ... rest of implementation
```

**Effort:** 2 hours

---

## 2. Server Architecture

### Framework Analysis

**Framework:** Python `http.server` (standard library)  
**Type:** Synchronous with `ThreadingHTTPServer`  
**Async Support:** None

#### 🟡 MEDIUM Issues

#### 2.1 Synchronous HTTP Server — Not Production-Grade
**File:** `tools/hybrid_brain.py` lines 945-960  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
```

`ThreadingHTTPServer` spawns a thread per request. Under load:
- Thread exhaustion (default max threads: system-dependent)
- No backpressure
- No graceful degradation
- No async I/O for I/O-bound operations (Qdrant, Ollama calls)

**Better alternatives:**
- **FastAPI + Uvicorn** — Async, production-ready, auto-docs
- **Flask + Gunicorn** — Battle-tested WSGI
- **aiohttp** — Pure async Python

**Migration effort:** 2-3 days to FastAPI

**Quick fix (stay with current framework):**
```python
from concurrent.futures import ThreadPoolExecutor

class LimitedThreadHTTPServer(ThreadingHTTPServer):
    def __init__(self, *args, max_workers=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread, request, client_address)
```

**Effort:** 4 hours for quick fix, 2-3 days for FastAPI migration

---

#### 2.2 No Connection Pooling for Downstream Services
**File:** `tools/hybrid_brain.py` lines 27-30  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
qdrant = QdrantClient(url=QDRANT_URL)
# Redis connection created fresh on every request
r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT)
```

**Every request creates new connections:**
- Qdrant: Uses `requests` (has internal pooling, but not configurable)
- Redis: New TCP connection per request → connection exhaustion

**Fix:**
```python
# Qdrant with connection pooling
from qdrant_client import QdrantClient
import urllib3

# Configure connection pool
http_adapter = urllib3.PoolManager(
    maxsize=50,  # Max connections
    block=True,
)
qdrant = QdrantClient(
    url=QDRANT_URL,
    http_adapter=http_adapter,
)

# Redis connection pool
import redis
redis_pool = redis.ConnectionPool(
    host=FALKOR_HOST,
    port=FALKOR_PORT,
    max_connections=50,
    decode_responses=False,
)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)
```

**Effort:** 2 hours

---

#### 2.3 No Graceful Shutdown
**File:** `tools/hybrid_brain.py` lines 945-960  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
def serve(port=7777):
    server = ReusableHTTPServer(("127.0.0.1", port), HybridHandler)
    server.serve_forever()
```

No signal handling. `SIGTERM`/`SIGINT` will:
- Kill active requests mid-flight
- Drop in-flight commits
- Leave Qdrant/FalkorDB in inconsistent state

**Fix:**
```python
import signal
import sys

def graceful_shutdown(signum, frame):
    print(f"[HybridBrain] Received signal {signum}, shutting down...", flush=True)
    server.shutdown()  # Wait for active requests
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

def serve(port=7777):
    server = ReusableHTTPServer(("0.0.0.0", port), HybridHandler)
    print(f"[HybridBrain] Serving on http://0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        graceful_shutdown(None, None)
```

**Effort:** 1 hour

---

## 3. Configuration

### Configuration Analysis

#### 🟡 MEDIUM Issues

#### 3.1 Hardcoded Constants Mixed with Environment Variables
**File:** `tools/hybrid_brain.py` lines 23-45  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# ... but then:
AMAC_THRESHOLD = 4.0  # ⚠️ Hardcoded!
AMAC_TIMEOUT = 30     # ⚠️ Hardcoded!
STOP_WORDS = {...}    # ⚠️ Hardcoded set!
```

**Inconsistent configuration:**
- Some values configurable via `.env`
- Critical thresholds hardcoded
- `STOP_WORDS` set defined in code (should be configurable for different languages)

**Fix:**
```python
# Add to .env.example
AMAC_THRESHOLD=4.0
AMAC_TIMEOUT=30
STOP_WORDS_FILE=config/stop_words.json

# In hybrid_brain.py
AMAC_THRESHOLD = float(os.environ.get("AMAC_THRESHOLD", "4.0"))
AMAC_TIMEOUT = int(os.environ.get("AMAC_TIMEOUT", "30"))

# Load STOP_WORDS from file
STOP_WORDS_FILE = os.environ.get("STOP_WORDS_FILE", "config/stop_words.json")
try:
    with open(STOP_WORDS_FILE) as f:
        STOP_WORDS = set(json.load(f))
except FileNotFoundError:
    STOP_WORDS = {...}  # Fallback defaults
```

**Effort:** 2 hours

---

#### 3.2 No Configuration Validation on Startup
**File:** Entire startup sequence  
**Severity:** 🟡 MEDIUM

**Problem:** Server starts even with invalid config:
```python
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# No validation — could be "not-a-url" or "ftp://broken"
```

**Fix:**
```python
def validate_config():
    errors = []
    
    # Validate URLs
    import re
    url_pattern = re.compile(r'^https?://[^\s]+$')
    if not url_pattern.match(QDRANT_URL):
        errors.append(f"Invalid QDRANT_URL: {QDRANT_URL}")
    
    if not url_pattern.match(EMBED_URL):
        errors.append(f"Invalid EMBED_URL: {EMBED_URL}")
    
    # Validate ports
    if not (1 <= FALKOR_PORT <= 65535):
        errors.append(f"Invalid FALKORDB_PORT: {FALKOR_PORT}")
    
    # Validate thresholds
    if not (0 <= AMAC_THRESHOLD <= 10):
        errors.append(f"AMAC_THRESHOLD must be 0-10, got {AMAC_THRESHOLD}")
    
    if errors:
        print("[HybridBrain] CONFIG ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

validate_config()  # Call before starting server
```

**Effort:** 2 hours

---

#### 3.3 Missing `.env` in Docker Image
**File:** `Dockerfile` lines 1-20  
**Severity:** 🟡 MEDIUM

**Problem:**
```dockerfile
ENV QDRANT_URL=http://host.docker.internal:6333
ENV FALKORDB_HOST=host.docker.internal
# ...
CMD ["python3", "tools/hybrid_brain.py"]
```

No `.env` file copying or mounting. Environment vars are **hardcoded in Dockerfile**. To change config, must rebuild image.

**Fix:**
```dockerfile
# Copy .env if present (for local dev)
COPY .env .env 2>/dev/null || true

# Or mount at runtime
# Volumes: - ./.env:/app/.env:ro

# Read from .env if exists
RUN pip install python-dotenv

CMD ["python3", "-c", "from dotenv import load_dotenv; load_dotenv(); import tools.hybrid_brain"]
```

**Effort:** 1 hour

---

#### 3.4 No Feature Flags for Optional Components
**File:** `tools/hybrid_brain.py` lines 45-50  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
FALKORDB_DISABLED = os.environ.get("DISABLE_FALKORDB", "0") == "1"
```

Only FalkorDB has a disable flag. What about:
- Reranker (can run without it)
- A-MAC gate
- Graph enrichment
- BM25 layer

**Fix:**
```python
# Feature flags
FEATURES = {
    'falkordb': os.environ.get("FEATURE_FALKORDB", "true") == "true",
    'reranker': os.environ.get("FEATURE_RERANKER", "true") == "true",
    'amac': os.environ.get("FEATURE_AMAC", "true") == "true",
    'bm25': os.environ.get("FEATURE_BM25", "true") == "true",
}

# Usage
if FEATURES['reranker'] and is_reranker_available():
    results = neural_rerank(query, results)
```

**Effort:** 2 hours

---

## 4. Docker Setup

### Dockerfile Analysis

#### 🔴 CRITICAL Issues

#### 4.1 Dockerfile Runs as Root
**File:** `Dockerfile` (entire file)  
**Severity:** 🔴 CRITICAL

**Problem:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
# ... no USER directive!
CMD ["python3", "tools/hybrid_brain.py"]
```

Container runs as **root user**. Security implications:
- If container compromised, attacker has root
- Can escape to host via volume mounts
- Violates least-privilege principle

**Fix:**
```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY tools/ tools/
COPY config/ config/

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root
USER appuser

ENV QDRANT_URL=http://host.docker.internal:6333
EXPOSE 7777

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:7777/health || exit 1

CMD ["python3", "tools/hybrid_brain.py"]
```

**Effort:** 1 hour

---

#### 4.2 No Multi-Stage Build
**File:** `Dockerfile` (entire file)  
**Severity:** 🟡 MEDIUM

**Problem:** Single-stage build includes:
- Build dependencies in final image
- `requirements.txt` copied unnecessarily
- Larger image size (security surface + transfer time)

**Fix:**
```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy only runtime dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY tools/ tools/
COPY config/ config/

# ... rest of Dockerfile
```

**Effort:** 1 hour

---

#### 4.3 No Image Size Optimization
**File:** `Dockerfile` (entire file)  
**Severity:** 🟡 MEDIUM

**Problem:** No size optimization:
- `apt-get` cache not cleaned (already fixed for curl)
- No `.dockerignore` for build context
- No layer optimization

**Current `.dockerignore`:** Good, excludes most unnecessary files

**Additional optimizations:**
```dockerfile
# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Use slim base (already using python:3.11-slim — good)

# Add layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tools/ tools/  # Changed order for better cache hits
```

**Effort:** 30 minutes

---

### docker-compose.yml Issues

#### 🟠 HIGH Issues

#### 4.4 No Resource Limits on Services
**File:** `docker-compose.yml` (entire file)  
**Severity:** 🟠 HIGH

**Problem:** No resource constraints:
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "127.0.0.1:6333:6333"
    # No memory/CPU limits!
```

Under load:
- Qdrant can consume all system memory
- Brain service can OOM killer other containers
- No fair resource sharing

**Fix:**
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '2.0'
          memory: 2G
    restart: unless-stopped

  brain:
    build: .
    ports:
      - "7777:7777"
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

**Effort:** 2 hours

---

#### 4.5 No Health Check for FalkorDB
**File:** `docker-compose.yml` lines 20-25  
**Severity:** 🟡 MEDIUM

**Problem:**
```yaml
falkordb:
  image: falkordb/falkordb:latest
  ports:
    - "127.0.0.1:6380:6379"
  volumes:
    - falkordb_data:/data
  restart: unless-stopped
  # NO HEALTH CHECK!
```

FalkorDB has no health check. `brain` service depends on it but:
- No way to detect FalkorDB is unresponsive
- `depends_on` only waits for container start, not readiness

**Fix:**
```yaml
falkordb:
  image: falkordb/falkordb:latest
  ports:
    - "127.0.0.1:6380:6379"
  volumes:
    - falkordb_data:/data
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "redis-cli", "-p", "6380", "ping"]
    interval: 10s
    timeout: 5s
    retries: 3
    start_period: 10s

brain:
  depends_on:
    qdrant:
      condition: service_healthy
    falkordb:  # Add this
      condition: service_healthy
    redis:
      condition: service_healthy
```

**Effort:** 30 minutes

---

#### 4.6 No Volume Persistence for Brain Data
**File:** `docker-compose.yml` (entire file)  
**Severity:** 🟡 MEDIUM

**Problem:** Brain service has no persistent volumes. If container restarts:
- In-memory state lost (acceptable for API server)
- But no way to persist logs, configs, or custom data

**Fix:**
```yaml
brain:
  build: .
  ports:
    - "7777:7777"
  volumes:
    - ./logs:/app/logs  # Persistent logs
    - ./config:/app/config:ro  # Config files
  # ... rest
```

**Effort:** 30 minutes

---

#### 4.7 No Network Isolation
**File:** `docker-compose.yml` (entire file)  
**Severity:** 🟡 MEDIUM

**Problem:** All services on default bridge network. No network segmentation:
- Brain can talk to Qdrant, FalkorDB, Redis (expected)
- But also can talk to anything else on host network
- No isolation between services

**Fix:**
```yaml
networks:
  memory_net:
    driver: bridge

services:
  qdrant:
    networks:
      - memory_net
  
  falkordb:
    networks:
      - memory_net
  
  redis:
    networks:
      - memory_net
  
  brain:
    networks:
      - memory_net
    # Can only reach services on memory_net
```

**Effort:** 1 hour

---

## 5. Health Checks

### Existing Health Check Analysis

#### 🟡 MEDIUM Issues

#### 5.1 Health Check Doesn't Validate All Dependencies
**File:** `tools/hybrid_brain.py` lines 800-835  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
elif parsed.path == "/health":
    health = {
        "status": "ok",
        "components": {
            "qdrant": "unknown",
            "falkordb": "unknown",
            "ollama_embed": "unknown",
            "reranker": "up" if is_reranker_available() else "down",
            "bm25": "up" if BM25_AVAILABLE else "down",
        }
    }
    # Checks Qdrant, FalkorDB, Ollama embed, reranker
    # BUT NOT: A-MAC LLM, graph enrichment, BM25 server
```

**Missing checks:**
- A-MAC LLM endpoint (`AMAC_LLM_URL`)
- BM25 reranker server
- Graph enrichment API (`GRAPH_API_URL`)
- Redis connection pool health

**Fix:**
```python
def check_amac_llm():
    try:
        resp = requests.post(
            os.environ.get("AMAC_LLM_URL", "http://localhost:11436/v1/chat/completions"),
            json={"model": "test", "messages": [{"role": "user", "content": "ping"}]},
            timeout=5
        )
        return "up" if resp.status_code == 200 else "error"
    except:
        return "down"

# Add to health endpoint
health["components"]["amac_llm"] = check_amac_llm()
```

**Effort:** 2 hours

---

#### 5.2 No Readiness vs Liveness Distinction
**File:** `/health` endpoint  
**Severity:** 🟡 MEDIUM

**Problem:** Single `/health` endpoint serves both:
- **Liveness:** Is the process alive? (for restarts)
- **Readiness:** Can it handle traffic? (for load balancing)

Kubernetes/docker-compose need different semantics:
- Liveness failure → restart container
- Readiness failure → remove from load balancer

**Fix:**
```python
# /health — liveness (always returns ok if process running)
elif parsed.path == "/health":
    self._send_json({"status": "ok", "uptime": time.time() - start_time})

# /ready — readiness (checks all dependencies)
elif parsed.path == "/ready":
    health = check_all_dependencies()  # Same as current /health
    status_code = 200 if health["status"] == "ok" else 503
    self._send_json(health, status_code)
```

**Effort:** 1 hour

---

#### 5.3 No Startup Probe for Slow Dependencies
**File:** Dockerfile healthcheck  
**Severity:** 🟡 MEDIUM

**Problem:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:7777/health || exit 1
```

No startup probe. If Ollama model is loading (can take 2-3 minutes):
- Health check fails immediately
- Container marked unhealthy
- May restart before model ready

**Fix:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
  CMD curl -f http://localhost:7777/ready || exit 1
```

**Effort:** 30 minutes

---

## 6. Error Handling

### Error Handling Analysis

#### 🟠 HIGH Issues

#### 6.1 No Global Exception Handler
**File:** `tools/hybrid_brain.py` HTTP handlers  
**Severity:** 🟠 HIGH

**Problem:** No try-catch around request handlers. Unhandled exceptions:
- Crash the thread
- Return 500 with stack trace (if not suppressed)
- No error logging

**Fix:**
```python
import traceback

def handle_request_safely(handler_func):
    @wraps(handler_func)
    def wrapper(self, *args, **kwargs):
        try:
            return handler_func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
            self._send_json({"error": "Internal server error"}, 500)
    return wrapper

class HybridHandler(BaseHTTPRequestHandler):
    @handle_request_safely
    def do_POST(self):
        # ... existing code
```

**Effort:** 3 hours

---

#### 6.2 No Timeout on Downstream Calls
**File:** `tools/hybrid_brain.py` — all `requests.post/get` calls  
**Severity:** 🟠 HIGH

**Problem:** Some calls have timeouts, some don't:
```python
# Has timeout
resp = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": prefixed_text}, timeout=35)

# No timeout!
resp = requests.get(f"{GRAPH_API_URL}/expand", params={"entity": name, "hops": 2}, timeout=2)

# No timeout!
r.execute_command('GRAPH.QUERY', GRAPH_NAME, "...")  # Redis call
```

Missing timeouts can:
- Hang indefinitely if downstream is unresponsive
- Block worker threads
- Cause cascade failures

**Fix:** Set consistent timeouts:
```python
TIMEOUTS = {
    'embedding': 35,
    'reranker': 15,
    'llm': 30,
    'qdrant': 10,
    'falkordb': 5,
    'graph_api': 2,
}

# Apply to all calls
requests.post(EMBED_URL, json=..., timeout=TIMEOUTS['embedding'])
```

**Effort:** 2 hours

---

#### 6.3 No Circuit Breaker for Downstream Services
**File:** Entire service — no circuit breaker pattern  
**Severity:** 🟠 HIGH

**Problem:** When Ollama/Qdrant/FalkorDB goes down:
- Every request fails
- No graceful degradation
- No fallback behavior

**Fix:**
```python
from circuitbreaker import CircuitBreaker

# Or implement simple version
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker open")
        
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            self.state = 'closed'
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.failure_threshold:
                self.state = 'open'
            raise

# Usage
embedding_cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
vector = embedding_cb.call(get_embedding, text)
```

**Effort:** 4 hours

---

#### 6.4 No Retry Logic with Exponential Backoff
**File:** `tools/hybrid_brain.py` lines 75-100  
**Severity:** 🟡 MEDIUM

**Problem:** Some functions have retry logic, but inconsistent:
```python
# Has retry
for attempt in range(max_retries):
    try:
        resp = requests.post(...)
    except Timeout:
        if attempt < max_retries - 1:
            time.sleep(2)  # ⚠️ Fixed delay, not exponential!

# No retry
results = qdrant.query_points(...)  # Just fails if Qdrant down
```

**Fix:**
```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries}: {e}")
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)  # Exponential backoff
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, base_delay=1.0)
def get_embedding_safe(text):
    return get_embedding(text)
```

**Effort:** 2 hours

---

#### 6.5 No Graceful Degradation When Components Fail
**File:** Search pipeline — all-or-nothing  
**Severity:** 🟡 MEDIUM

**Problem:** If reranker is down, search still works but quality drops. However:
- No indication to client that quality degraded
- No fallback ranking strategy
- No caching of previous results

**Fix:**
```python
def hybrid_search(query, limit=10, graph_hops=2, source_filter=None):
    # ... existing code
    
    # Check component health
    reranker_up = is_reranker_available()
    if not reranker_up:
        logger.warning("Reranker down — returning degraded results")
        # Add header to response
        result["degraded"] = True
        result["missing_components"] = ["reranker"]
    
    return result
```

**Effort:** 2 hours

---

## 7. Logging & Observability

### Logging Analysis

#### 🔴 CRITICAL Issues

#### 7.1 No Structured Logging
**File:** `tools/hybrid_brain.py` — all `print()` statements  
**Severity:** 🔴 CRITICAL

**Problem:**
```python
print("[HybridBrain] BM25 reranking: enabled", flush=True)
print(f"[Embedding] Timeout on attempt {attempt+1}/{max_retries}", flush=True)
```

**All logging is unstructured `print()` statements:**
- Cannot parse programmatically
- No log levels (INFO, WARN, ERROR)
- No correlation IDs for tracing
- Cannot filter by component/severity
- Cannot ship to log aggregation (ELK, Datadog, etc.)

**Fix:**
```python
import logging
import json
from datetime import datetime

# Configure structured logger
logger = logging.getLogger("hybrid_brain")
logger.setLevel(logging.INFO)

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "component": record.name,
            "module": record.module,
            "line": record.lineno,
        }
        # Add extra fields if present
        if hasattr(record, 'client_ip'):
            log_entry["client_ip"] = record.client_ip
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        return json.dumps(log_entry)

handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)

# Usage
logger.info("Request received", extra={"client_ip": client_ip, "request_id": req_id})
logger.error("Embedding failed", extra={"error": str(e), "text_length": len(text)})
```

**Effort:** 4 hours

---

#### 7.2 No Request Correlation/Tracing
**File:** Entire codebase — no tracing  
**Severity:** 🟠 HIGH

**Problem:** No way to trace a request across:
- HTTP → embedding → Qdrant → reranker → response
- Cannot identify bottleneck
- Cannot debug multi-component failures

**Fix:** Add correlation IDs:
```python
import uuid

class HybridHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info("Request started", extra={
            "request_id": request_id,
            "path": self.path,
            "method": self.command,
        })
        
        try:
            # ... process request
            result = hybrid_search(query, limit=limit)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info("Request completed", extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "status": "success",
            })
            
            self._send_json(result)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Request failed", extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "error": str(e),
            })
            self._send_json({"error": str(e)}, 500)
```

**Effort:** 3 hours

---

#### 7.3 No Metrics/Monitoring
**File:** Entire codebase — no metrics  
**Severity:** 🟠 HIGH

**Problem:** No metrics exposed:
- Request rate (RPS)
- Error rate
- Latency percentiles (p50, p95, p99)
- Memory/CPU usage
- Queue depths

**Fix:** Add Prometheus metrics:
```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
REQUEST_COUNT = Counter('hybrid_brain_requests_total', 'Total requests', ['method', 'path', 'status'])
REQUEST_DURATION = Histogram('hybrid_brain_request_duration_seconds', 'Request duration')
SEARCH_RESULTS = Histogram('hybrid_brain_search_results_count', 'Number of search results')
AMAC_SCORES = Histogram('hybrid_brain_amac_scores', 'A-MAC admission scores')

# In handler
@REQUEST_DURATION.time()
def do_POST(self):
    REQUEST_COUNT.labels(method='POST', path=self.path, status=200).inc()
    # ... process
```

**Effort:** 6 hours

---

#### 7.4 No Log Rotation
**File:** All logging — unbounded  
**Severity:** 🟡 MEDIUM

**Problem:** Logs grow forever:
- Disk fills up
- Logs become slow to search
- No retention policy

**Fix:**
```python
from logging.handlers import RotatingFileHandler

# Rotate logs at 100MB, keep 5 files
handler = RotatingFileHandler(
    'logs/hybrid_brain.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=5,
)
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
```

**Effort:** 1 hour

---

## 8. Security

### Security Analysis

#### 🔴 CRITICAL Issues

#### 8.1 No Input Sanitization for Cypher Queries
**File:** `tools/hybrid_brain.py` lines 220-245  
**Severity:** 🔴 CRITICAL

**Problem:**
```python
# Entity label whitelist (GOOD)
safe_label = etype if etype in ("Person", "Organization", "Project", "Topic", "Location") else "Entity"
r.execute_command('GRAPH.QUERY', GRAPH_NAME,
    f"MERGE (n:{safe_label} {{name: $name}}) "  # ⚠️ f-string with user input!
    f"ON CREATE SET n.type = $etype, n.created_at = $ts "
    f"WITH n MATCH (m:Memory {{id: $id}}) MERGE (n)-[:MENTIONED_IN]->(m)",
    '--params', json.dumps({"name": name, "etype": etype, "ts": ts, "id": str(point_id)}))
```

**Cypher injection risk:** While `safe_label` is whitelisted, the `name` parameter is passed via `--params`. However, if an attacker can control the entity name:
- Special characters in names could break query syntax
- Newlines could inject additional Cypher commands

**Fix:** Sanitize entity names:
```python
def sanitize_entity_name(name: str) -> str:
    """Sanitize entity name for Cypher queries."""
    # Remove/escape dangerous characters
    name = name.replace("'", "\\'")  # Escape single quotes
    name = name.replace("\n", " ")   # Remove newlines
    name = name.replace("\r", "")    # Remove carriage returns
    name = name.replace(";", "")     # Remove semicolons (statement separator)
    return name[:100]  # Limit length

# Usage
safe_name = sanitize_entity_name(name)
r.execute_command('GRAPH.QUERY', GRAPH_NAME,
    f"MERGE (n:{safe_label} {{name: $name}}) ...",
    '--params', json.dumps({"name": safe_name, ...}))
```

**Effort:** 2 hours

---

#### 8.2 No Rate Limiting on Health Endpoint
**File:** `/health` endpoint  
**Severity:** 🟡 MEDIUM

**Problem:** Health endpoint has no rate limit. Attacker can:
- Flood with health checks
- Exhaust server resources
- Cause denial of service

**Fix:** Add rate limiting to all endpoints (see 1.1)

**Effort:** Already covered in 1.1

---

#### 8.3 No CSRF Protection
**File:** POST endpoints  
**Severity:** 🟡 MEDIUM

**Problem:** No CSRF tokens on POST endpoints. If deployed with browser access:
- Cross-site request forgery possible
- Attacker can submit memories via malicious site

**Fix:**
```python
# Add CSRF token validation
CSRF_SECRET = os.environ.get("CSRF_SECRET", os.urandom(32).hex())

def generate_csrf_token():
    import hmac
    return hmac.new(CSRF_SECRET, str(time.time()).encode(), 'sha256').hexdigest()

def validate_csrf_token(token):
    # Validate token signature/timestamp
    pass

class HybridHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Check CSRF token for state-changing operations
        csrf_token = self.headers.get("X-CSRF-Token")
        if not validate_csrf_token(csrf_token):
            self._send_json({"error": "Invalid CSRF token"}, 403)
            return
```

**Effort:** 3 hours

---

#### 8.4 No Content-Type Validation
**File:** POST handlers  
**Severity:** 🟡 MEDIUM

**Problem:**
```python
def do_POST(self):
    content_length = int(self.headers.get("Content-Length", 0))
    body = self.rfile.read(content_length)
    try:
        data = json.loads(body) if body else {}
    except:
        self._send_json({"error": "Invalid JSON"}, 400)
        return
```

No `Content-Type` header validation. Accepts:
- `text/plain` with JSON body
- `application/xml` (will fail JSON parse but still processed)
- Any content type

**Fix:**
```python
content_type = self.headers.get("Content-Type", "")
if "application/json" not in content_type:
    self._send_json({"error": "Content-Type must be application/json"}, 415)
    return
```

**Effort:** 30 minutes

---

### 🟠 HIGH Issues

#### 8.5 Sensitive Data in Logs
**File:** All logging statements  
**Severity:** 🟠 HIGH

**Problem:** Logs may contain sensitive data:
- Memory text (could contain PII, secrets, business data)
- API tokens (if passed in headers)
- User identifiers

**Fix:** Add log sanitization:
```python
SENSITIVE_PATTERNS = [
    r'api[_-]?key',
    r'token',
    r'password',
    r'secret',
    r'Bearer\s+[a-zA-Z0-9\-_]+',
]

def sanitize_for_logging(text: str) -> str:
    for pattern in SENSITIVE_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
    return text

logger.info(f"Request text: {sanitize_for_logging(text[:200])}")
```

**Effort:** 2 hours

---

#### 8.6 No IP Whitelisting/Blacklisting
**File:** Entire server — no IP filtering  
**Severity:** 🟠 HIGH

**Problem:** No access control by IP. If exposed to internet:
- Anyone can query/search
- No geo-blocking
- No IP-based rate limiting

**Fix:**
```python
ALLOWED_IPS = os.environ.get("ALLOWED_IPS", "").split(",")
BLOCKED_IPS = os.environ.get("BLOCKED_IPS", "").split(",")

def _check_ip(self):
    client_ip = self.client_address[0]
    if client_ip in BLOCKED_IPS:
        self._send_json({"error": "Forbidden"}, 403)
        return False
    if ALLOWED_IPS and client_ip not in ALLOWED_IPS:
        self._send_json({"error": "Forbidden"}, 403)
        return False
    return True
```

**Effort:** 2 hours

---

## 9. quickstart.sh Analysis

### Script Quality Assessment

#### 🟡 MEDIUM Issues

#### 9.1 No Idempotency Checks
**File:** `quickstart.sh` (entire script)  
**Severity:** 🟡 MEDIUM

**Problem:** Script is mostly idempotent but has gaps:
```bash
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example"
else
    echo "📝 .env already exists — keeping your settings"
fi
```

Good: `.env` creation is idempotent

Bad: No check if Docker services are already running
```bash
echo "🐳 Starting Qdrant + FalkorDB + Redis..."
$COMPOSE up -d  # ⚠️ Will fail if already running (or succeed silently)
```

**Fix:**
```bash
# Check if services already running
if $COMPOSE ps | grep -q "Up"; then
    echo "⚠️  Services already running. Stopping first..."
    $COMPOSE down
fi

$COMPOSE up -d
```

**Effort:** 1 hour

---

#### 9.2 No Error Handling for Ollama Installation
**File:** `quickstart.sh` lines 55-65  
**Severity:** 🟡 MEDIUM

**Problem:**
```bash
if ! command -v ollama >/dev/null 2>&1; then
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  ✅ Ollama already installed"
fi
```

No verification that Ollama installation succeeded. Script continues even if:
- Download fails
- Installation fails
- Service doesn't start

**Fix:**
```bash
if ! command -v ollama >/dev/null 2>&1; then
    echo "📥 Installing Ollama..."
    if ! curl -fsSL https://ollama.com/install.sh | sh; then
        echo "❌ Ollama installation failed"
        exit 1
    fi
    
    # Verify installation
    if ! ollama --version >/dev/null 2>&1; then
        echo "❌ Ollama installation verification failed"
        exit 1
    fi
    echo "  ✅ Ollama installed successfully"
else
    echo "  ✅ Ollama already installed (version: $(ollama --version))"
fi
```

**Effort:** 1 hour

---

#### 9.3 No Cleanup on Failure
**File:** `quickstart.sh` (entire script)  
**Severity:** 🟡 MEDIUM

**Problem:** If script fails mid-way:
```bash
$COMPOSE up -d  # Started services
# ... script fails here
pip install -r requirements.txt  # Never runs
```

Leaves:
- Docker containers running
- Partial state
- User must manually clean up

**Fix:**
```bash
#!/bin/bash
set -e  # Already present

# Add cleanup trap
cleanup() {
    echo "⚠️  Setup failed. Cleaning up..."
    $COMPOSE down -v 2>/dev/null || true
    rm -f .env 2>/dev/null || true
}
trap cleanup EXIT

# ... rest of script
```

**Effort:** 1 hour

---

#### 9.4 No Version Pinning for Dependencies
**File:** `quickstart.sh` — uses `requirements.txt`  
**Severity:** 🟡 MEDIUM

**Problem:**
```bash
pip install -q -r requirements.txt
```

`requirements.txt` has loose versioning:
```
qdrant-client>=1.9.0
redis>=5.0.0
```

Can install incompatible versions. Should pin exact versions for reproducibility.

**Fix:**
```bash
# requirements.txt
qdrant-client==1.9.0
redis==5.0.0
FalkorDB==1.0.0
requests==2.31.0
python-dotenv==1.0.0
```

**Effort:** 30 minutes

---

#### 9.5 No Post-Setup Verification
**File:** `quickstart.sh` end  
**Severity:** 🟡 MEDIUM

**Problem:** Script ends with:
```bash
echo "🎉 Setup complete!"
echo ""
echo "Start the memory server:"
echo "  python3 tools/hybrid_brain.py"
```

No verification that setup actually works. User must manually test.

**Fix:**
```bash
echo "🎉 Setup complete!"
echo ""
echo "Running verification tests..."

# Test Qdrant
if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
    echo "  ✅ Qdrant responding"
else
    echo "  ❌ Qdrant not responding"
    exit 1
fi

# Test Redis
if $COMPOSE exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "  ✅ Redis responding"
else
    echo "  ⚠️  Redis not responding (graph features unavailable)"
fi

echo ""
echo "✅ All checks passed!"
echo "Setup complete!"
```

**Effort:** 2 hours

---

#### 9.6 No Support for Non-Default Paths
**File:** `quickstart.sh` (entire script)  
**Severity:** 🔵 LOW

**Problem:** Hardcoded paths:
```bash
mkdir -p logs memory/hot-context
```

No option to specify custom workspace or config paths.

**Fix:**
```bash
WORKSPACE=${WORKSPACE:-$(pwd)}
cd "$WORKSPACE"

mkdir -p logs memory/hot-context
```

**Effort:** 30 minutes

---

## 10. Recommendations Summary

### Priority 1: Critical Fixes (Do Immediately)

| Issue | Severity | Effort | Impact |
|-------|----------|--------|--------|
| Add rate limiting | 🔴 CRITICAL | 2-3h | Prevents DoS |
| Enable auth by default | 🔴 CRITICAL | 1h | Security baseline |
| Run Docker as non-root | 🔴 CRITICAL | 1h | Container security |
| Add input validation | 🔴 CRITICAL | 2h | Prevent injection |
| Structured logging | 🔴 CRITICAL | 4h | Observability |
| Sanitize Cypher inputs | 🔴 CRITICAL | 2h | Injection prevention |

**Total: ~12 hours**

---

### Priority 2: High-Value Improvements (Next Sprint)

| Issue | Severity | Effort | Impact |
|-------|----------|--------|--------|
| Request size limits | 🟠 HIGH | 30m | DoS prevention |
| CORS configuration | 🟠 HIGH | 1h | Browser safety |
| HTTP timeouts | 🟠 HIGH | 1h | Connection safety |
| TLS/HTTPS support | 🟠 HIGH | 2h | Data encryption |
| Global exception handler | 🟠 HIGH | 3h | Stability |
| Downstream timeouts | 🟠 HIGH | 2h | Cascade prevention |
| Circuit breaker pattern | 🟠 HIGH | 4h | Resilience |
| Sensitive data logging | 🟠 HIGH | 2h | Privacy |
| IP filtering | 🟠 HIGH | 2h | Access control |

**Total: ~19 hours**

---

### Priority 3: Production Readiness (Future Sprints)

| Issue | Severity | Effort | Impact |
|-------|----------|--------|--------|
| API versioning | 🟡 MEDIUM | 4h | Maintainability |
| Request logging/metrics | 🟡 MEDIUM | 5h | Observability |
| Connection pooling | 🟡 MEDIUM | 2h | Performance |
| Graceful shutdown | 🟡 MEDIUM | 1h | Reliability |
| Config validation | 🟡 MEDIUM | 2h | Startup safety |
| Resource limits (Docker) | 🟡 MEDIUM | 2h | Stability |
| Health check improvements | 🟡 MEDIUM | 3h | Kubernetes readiness |
| Structured logging setup | 🟡 MEDIUM | 4h | Debugging |
| Prometheus metrics | 🟡 MEDIUM | 6h | Monitoring |
| Log rotation | 🟡 MEDIUM | 1h | Disk safety |

**Total: ~30 hours**

---

### Priority 4: Nice-to-Have (Backlog)

| Issue | Severity | Effort | Impact |
|-------|----------|--------|--------|
| Multi-stage Docker build | 🔵 LOW | 1h | Image size |
| Docker network isolation | 🔵 LOW | 1h | Security |
| Feature flags | 🔵 LOW | 2h | Flexibility |
| CSRF protection | 🔵 LOW | 3h | Browser safety |
| Content-Type validation | 🔵 LOW | 30m | API hygiene |
| quickstart.sh improvements | 🔵 LOW | 4h | UX |

**Total: ~11.5 hours**

---

## 11. Overall Architecture Assessment

### Strengths

1. **Hybrid search pipeline** — Well-designed combination of vector + graph + keyword + neural reranking
2. **Graceful degradation** — System works without reranker, graph, or even A-MAC
3. **A-MAC quality gate** — Innovative approach to memory quality control
4. **BM25 layer** — Good addition for keyword matching on top of vectors
5. **Temporal decay** — Ebbinghaus forgetting curve implemented correctly
6. **Health checks** — Basic component monitoring in place
7. **Docker Compose** — Reasonable infrastructure-as-code setup

### Weaknesses

1. **Security posture** — No auth by default, no input validation, runs as root
2. **Production readiness** — Not suitable for production deployment without significant work
3. **Observability** — Logging is `print()` statements, no metrics, no tracing
4. **Error handling** — Inconsistent timeouts, no circuit breakers, no retry logic
5. **Scalability** — Single-threaded HTTP server, no connection pooling
6. **Configuration** — Hardcoded constants mixed with env vars, no validation
7. **Documentation** — Good docs but missing security/operational runbooks

### Recommendations

1. **Immediate:** Implement critical security fixes (auth, rate limiting, input validation)
2. **Short-term:** Add structured logging, metrics, and proper error handling
3. **Medium-term:** Migrate to FastAPI for async support and better production features
4. **Long-term:** Consider Kubernetes deployment with proper HPA, service mesh, and observability stack

---

## 12. Appendix: Security Checklist

- [x] **Authentication:** Optional token-based (needs to be mandatory)
- [ ] **Authorization:** None (no role-based access control)
- [x] **Rate Limiting:** None (needs implementation)
- [x] **Input Validation:** Minimal (needs comprehensive validation)
- [x] **Output Encoding:** None (potential XSS in rendered outputs)
- [x] **CORS:** Not configured (needs explicit policy)
- [x] **CSRF:** Not protected (needs tokens for browser access)
- [x] **TLS/HTTPS:** Not enabled (plaintext HTTP)
- [x] **Secrets Management:** Environment variables only (no vault/encryption)
- [x] **Dependency Scanning:** Not configured (needs automated scanning)
- [x] **Container Security:** Runs as root (needs non-root user)
- [x] **Network Segmentation:** No isolation (needs Docker network policies)
- [x] **Logging:** Unstructured print statements (needs structured logging)
- [x] **Audit Trail:** None (needs request logging)
- [x] **Incident Response:** No runbooks (needs operational documentation)

**Security Score: 3/15 (20%)**

---

**Report Generated:** 2026-03-30  
**Auditor:** PhD Computer Scientist (Infrastructure/Security Domain)  
**Classification:** Internal Use Only
