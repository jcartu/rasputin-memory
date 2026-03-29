FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the brain server needs
COPY tools/ tools/
COPY config/ config/

# Environment defaults
ENV QDRANT_URL=http://host.docker.internal:6333
ENV FALKORDB_HOST=host.docker.internal
ENV REDIS_HOST=host.docker.internal
ENV OLLAMA_URL=http://host.docker.internal:11434
ENV PORT=7777

EXPOSE 7777

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:7777/health || exit 1

CMD ["python3", "tools/hybrid_brain.py"]
