FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "fastmcp>=3.2.0"

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY tools/mcp/ tools/mcp/

RUN chown -R appuser:appgroup /app

ENV RASPUTIN_URL=http://host.docker.internal:7777
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8808

EXPOSE 8808

USER appuser

CMD ["python3", "tools/mcp/server.py"]
