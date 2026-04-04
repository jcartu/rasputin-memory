"""Pluggable embedding providers: ollama (default), gemini, cohere."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_DIMS = int(os.environ.get("GEMINI_EMBED_DIMS", "768"))


def get_embedding_gemini(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    dims: int = GEMINI_DIMS,
) -> list[float]:
    """Embed text using Gemini Embedding 001 via Google AI API."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    )
    body = json.dumps(
        {
            "content": {"parts": [{"text": text[:8000]}]},
            "taskType": task_type,
            "outputDimensionality": dims,
        }
    ).encode()

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            values: list[float] = data["embedding"]["values"]
            # Gemini only normalizes at 3072d — normalize at other dims
            if dims != 3072:
                mag = math.sqrt(sum(v * v for v in values))
                if mag > 0:
                    values = [v / mag for v in values]
            return values
        except Exception as e:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Gemini embedding failed after retries: {e}")
    raise RuntimeError("Gemini embedding failed: exhausted retries")


def get_embedding_auto(text: str, prefix: str = "") -> list[float]:
    """Route to the configured embedding provider."""
    if EMBED_PROVIDER == "gemini" and GEMINI_API_KEY:
        # Map nomic-style prefixes to Gemini task types
        if "search_query" in prefix or "query" in prefix.lower():
            task_type = "RETRIEVAL_QUERY"
        elif "search_document" in prefix or "document" in prefix.lower():
            task_type = "RETRIEVAL_DOCUMENT"
        else:
            task_type = "RETRIEVAL_DOCUMENT"
        return get_embedding_gemini(text, task_type=task_type, dims=GEMINI_DIMS)
    else:
        from brain.embedding import _get_embedding_ollama

        return _get_embedding_ollama(text, prefix=prefix)
