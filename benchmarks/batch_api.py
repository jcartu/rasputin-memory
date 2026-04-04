"""Raw HTTP batch API clients for Anthropic Message Batches and OpenAI Batch API.

50% cost reduction on both. No SDK dependencies — pure urllib.request.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ─── Anthropic Message Batches ───────────────────────────────


def anthropic_create_batch(
    requests: list[dict[str, Any]],
    api_key: str = "",
) -> dict[str, Any]:
    """Submit a batch of message requests to Anthropic.

    Each request: {"custom_id": "...", "params": {"model": "...", "max_tokens": N, "messages": [...]}}
    Returns batch metadata including 'id' and 'processing_status'.
    """
    key = api_key or ANTHROPIC_API_KEY
    body = json.dumps({"requests": requests}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages/batches",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")

    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def anthropic_get_batch(batch_id: str, api_key: str = "") -> dict[str, Any]:
    key = api_key or ANTHROPIC_API_KEY
    req = urllib.request.Request(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
        method="GET",
    )
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def anthropic_poll_batch(
    batch_id: str,
    api_key: str = "",
    poll_interval: int = 60,
    max_wait: int = 7200,
) -> dict[str, Any]:
    """Poll until batch reaches 'ended' status. Returns final batch metadata."""
    key = api_key or ANTHROPIC_API_KEY
    deadline = time.time() + max_wait
    while time.time() < deadline:
        batch = anthropic_get_batch(batch_id, api_key=key)
        status = batch.get("processing_status", "")
        counts = batch.get("request_counts", {})
        succeeded = counts.get("succeeded", 0)
        total = sum(counts.get(k, 0) for k in ["processing", "succeeded", "errored", "canceled", "expired"])
        print(f"  Anthropic batch {batch_id[:16]}... status={status} ({succeeded}/{total} succeeded)")

        if status == "ended":
            return batch
        time.sleep(poll_interval)

    raise TimeoutError(f"Anthropic batch {batch_id} did not complete within {max_wait}s")


def anthropic_get_results(batch_id: str, api_key: str = "") -> list[dict[str, Any]]:
    """Download results JSONL from a completed batch."""
    key = api_key or ANTHROPIC_API_KEY
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
    req = urllib.request.Request(url, method="GET")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")

    results = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode().strip()
            if line:
                results.append(json.loads(line))
    return results


# ─── OpenAI Batch API ────────────────────────────────────────


def openai_upload_jsonl(jsonl_content: str, api_key: str = "") -> str:
    """Upload JSONL string to OpenAI Files API. Returns file_id."""
    key = api_key or OPENAI_API_KEY
    boundary = "----BatchUploadBoundary7MA4YWxk"
    body_parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n',
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
        f"Content-Type: application/jsonl\r\n\r\n"
        f"{jsonl_content}\r\n",
        f"--{boundary}--\r\n",
    ]
    body = "".join(body_parts).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/files",
        data=body,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["id"]


def openai_create_batch(
    input_file_id: str,
    api_key: str = "",
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create an OpenAI batch from an uploaded JSONL file."""
    key = api_key or OPENAI_API_KEY
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    if metadata:
        payload["metadata"] = metadata

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/batches",
        data=body,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def openai_get_batch(batch_id: str, api_key: str = "") -> dict[str, Any]:
    key = api_key or OPENAI_API_KEY
    req = urllib.request.Request(
        f"https://api.openai.com/v1/batches/{batch_id}",
        method="GET",
    )
    req.add_header("Authorization", f"Bearer {key}")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def openai_poll_batch(
    batch_id: str,
    api_key: str = "",
    poll_interval: int = 30,
    max_wait: int = 86400,
) -> dict[str, Any]:
    """Poll until batch reaches terminal status. Returns final batch metadata."""
    key = api_key or OPENAI_API_KEY
    deadline = time.time() + max_wait
    while time.time() < deadline:
        batch = openai_get_batch(batch_id, api_key=key)
        status = batch.get("status", "")
        counts = batch.get("request_counts", {})
        completed = counts.get("completed", 0)
        total = counts.get("total", 0)
        print(f"  OpenAI batch {batch_id[:16]}... status={status} ({completed}/{total} completed)")

        if status == "completed":
            return batch
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"OpenAI batch {status}: {batch}")
        time.sleep(poll_interval)

    raise TimeoutError(f"OpenAI batch {batch_id} did not complete within {max_wait}s")


def openai_get_results(output_file_id: str, api_key: str = "") -> list[dict[str, Any]]:
    """Download results JSONL from a completed batch's output file."""
    key = api_key or OPENAI_API_KEY
    req = urllib.request.Request(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        method="GET",
    )
    req.add_header("Authorization", f"Bearer {key}")

    results = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode().strip()
            if line:
                results.append(json.loads(line))
    return results
