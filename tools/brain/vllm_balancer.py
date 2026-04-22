"""Round-robin vLLM extraction load balancer.

Fronts two Qwen3-32B-AWQ instances (instance A on :11437, instance B on :11439)
and exposes a single endpoint on :11440 with identical OpenAI-compatible API.

Per user addendum 2026-04-21 "PARALLEL GPU TOPOLOGY":
- Round-robin between both instances
- /health failover: skip instance if /v1/models 5xx'd or timed out in last 30s
- Passthrough /v1/chat/completions and /v1/models unchanged
- Return 503 if both down

Port deviation: addendum specified :11438 for instance B, but nginx has that
port. Using :11439 for instance B. Update baseline_env.sh accordingly.
"""
from __future__ import annotations

import asyncio
import datetime
import itertools
import logging
import os
import time
from typing import Any

AUDIT_LOG = os.environ.get("VLLM_BALANCER_AUDIT_LOG", "/tmp/bench_runs/balancer.log")
_audit_fd: int | None = None


def _audit(instance: str, latency_ms: float, status: int) -> None:
    """Append one line per request to the audit log for routing verification."""
    global _audit_fd
    try:
        if _audit_fd is None:
            _audit_fd = os.open(AUDIT_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        line = f"{ts} instance={instance} status={status} latency_ms={latency_ms:.0f}\n"
        os.write(_audit_fd, line.encode())
    except OSError:
        pass  # audit is best-effort, never block requests

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s","logger":"vllm_balancer"}',
)
logger = logging.getLogger("vllm_balancer")

INSTANCES = [
    os.environ.get("VLLM_INSTANCE_A", "http://127.0.0.1:11437"),
    os.environ.get("VLLM_INSTANCE_B", "http://127.0.0.1:11439"),
]
HEALTH_TTL_SEC = 30
REQ_TIMEOUT_SEC = 600.0  # extraction calls can be slow under load

app = FastAPI(title="vllm-balancer", version="0.1.0")
_round_robin = itertools.cycle(INSTANCES)
_health: dict[str, dict[str, float | bool]] = {
    url: {"healthy": True, "last_check": 0.0} for url in INSTANCES
}
_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(REQ_TIMEOUT_SEC, connect=5.0))
    return _client


async def _check_health(url: str) -> bool:
    client = await _get_client()
    try:
        resp = await client.get(f"{url}/v1/models", timeout=httpx.Timeout(5.0))
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def _refresh_health(url: str) -> bool:
    """Check health if last check was >HEALTH_TTL_SEC ago. Return current state."""
    now = time.monotonic()
    entry = _health[url]
    if now - float(entry["last_check"]) > HEALTH_TTL_SEC:
        healthy = await _check_health(url)
        _health[url] = {"healthy": healthy, "last_check": now}
        if not healthy:
            logger.warning(f"instance {url} unhealthy")
    return bool(_health[url]["healthy"])


async def _pick_instance() -> str | None:
    """Round-robin, skipping unhealthy instances. None if both down."""
    for _ in range(len(INSTANCES)):
        candidate = next(_round_robin)
        if await _refresh_health(candidate):
            return candidate
    return None


@app.get("/health")
async def health() -> dict[str, Any]:
    states = {url: await _refresh_health(url) for url in INSTANCES}
    healthy_count = sum(1 for v in states.values() if v)
    return {
        "balancer": "ok",
        "instances": states,
        "healthy_count": healthy_count,
        "status": "ok" if healthy_count >= 1 else "degraded",
    }


@app.get("/v1/models")
async def models() -> JSONResponse:
    instance = await _pick_instance()
    if instance is None:
        raise HTTPException(status_code=503, detail="no healthy vLLM instances")
    client = await _get_client()
    resp = await client.get(f"{instance}/v1/models")
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse:
    instance = await _pick_instance()
    if instance is None:
        raise HTTPException(status_code=503, detail="no healthy vLLM instances")

    body = await request.body()
    headers = {"Content-Type": "application/json"}
    client = await _get_client()

    _t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{instance}/v1/chat/completions",
            content=body,
            headers=headers,
        )
    except httpx.HTTPError as e:
        # Mark instance unhealthy and retry once on the other instance
        _health[instance] = {"healthy": False, "last_check": time.monotonic()}
        logger.warning(f"instance {instance} request failed: {e!r}, retrying other")
        fallback = await _pick_instance()
        if fallback is None or fallback == instance:
            _audit(instance, (time.monotonic() - _t0) * 1000, 503)
            raise HTTPException(status_code=503, detail="no healthy vLLM instances") from e
        instance = fallback  # track which instance actually served
        resp = await client.post(
            f"{fallback}/v1/chat/completions",
            content=body,
            headers=headers,
        )

    # Non-streaming passthrough (fact extraction is always non-streaming)
    _audit(instance, (time.monotonic() - _t0) * 1000, resp.status_code)
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.on_event("shutdown")
async def shutdown() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("VLLM_BALANCER_PORT", "11440"))
    host = os.environ.get("VLLM_BALANCER_HOST", "127.0.0.1")
    logger.info(f"starting vllm-balancer on {host}:{port}, instances={INSTANCES}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
