"""Pluggable reranker providers: local BGE (default), cohere, llm."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")
RERANK_PROVIDER = os.environ.get("RERANK_PROVIDER", "cohere" if COHERE_API_KEY else "local")


def rerank_cohere(query: str, results: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    """Rerank using Cohere Rerank v3.5 — best-in-class neural reranker."""
    if not results or not COHERE_API_KEY:
        return results[:top_k]

    documents = []
    for r in results[:100]:
        text = (r.get("text") or "")[:4096]
        source = r.get("source", "")
        date = r.get("date", "")
        documents.append(f"[{source}, {date}] {text}" if source else text)

    body = json.dumps(
        {
            "model": COHERE_RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "return_documents": False,
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.cohere.com/v2/rerank",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {COHERE_API_KEY}")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            reranked: list[dict[str, Any]] = []
            for item in data.get("results", []):
                idx = item["index"]
                if idx < len(results):
                    row = results[idx].copy()
                    row["rerank_score"] = item["relevance_score"]
                    row["reranker"] = "cohere"
                    reranked.append(row)

            return reranked[:top_k] if reranked else results[:top_k]
        except Exception:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            return results[:top_k]
    return results[:top_k]
