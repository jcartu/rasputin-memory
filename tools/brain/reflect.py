"""Reflect -- synthesize answers from retrieved memories.

Unlike raw search (which returns ranked chunks), reflect runs an LLM
over the retrieved memories to produce a coherent, reasoned answer.
This is the equivalent of Hindsight's ``reflect`` operation.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional

from brain import _state
from brain import search

_reflect_cfg: dict[str, Any] = _state.CONFIG.get("reflect", {})

REFLECT_MODEL = os.environ.get("REFLECT_MODEL", _reflect_cfg.get("model", "claude-haiku-4-5-20251001"))
REFLECT_PROVIDER = os.environ.get("REFLECT_PROVIDER", _reflect_cfg.get("provider", "anthropic"))
REFLECT_MAX_TOKENS = int(os.environ.get("REFLECT_MAX_TOKENS", str(_reflect_cfg.get("max_tokens", 1000))))

_REFLECT_PROMPT = """\
You are answering a question using retrieved memories from a long-term memory system.
Synthesize a clear, specific answer from the evidence provided.  Connect dots across
multiple memories where relevant.  If memories contain contradictory information, note
the contradiction and favor the most recent memory.

If the memories don't contain enough information to answer, say so clearly -- do not fabricate.

Memories:
{context}

Question: {query}

Answer:"""


def reflect(
    query: str,
    limit: int = 20,
    source_filter: Optional[str] = None,
    collection: Optional[str] = None,
) -> dict[str, Any]:
    """Search memories and synthesize an answer via LLM.

    Returns::

        {
            "answer": "synthesized text",
            "sources": [{"point_id": ..., "text": ..., "score": ...}, ...],
            "search_elapsed_ms": float,
            "total_elapsed_ms": float,
            "reflect_model": str,
        }
    """
    start = time.time()

    # -- 1. Retrieve ---------------------------------------------------------
    search_result = search.hybrid_search(
        query,
        limit=limit,
        source_filter=source_filter,
        collection=collection,
    )
    results = search_result.get("results", [])
    search_ms = search_result.get("elapsed_ms", 0)

    if not results:
        return {
            "answer": "I don't have any relevant memories to answer this question.",
            "sources": [],
            "search_elapsed_ms": search_ms,
            "total_elapsed_ms": round((time.time() - start) * 1000, 1),
            "reflect_model": REFLECT_MODEL,
        }

    # -- 2. Format context (cap at 15 for prompt length) --------------------
    context_parts: list[str] = []
    for i, r in enumerate(results[:15]):
        date = (r.get("date") or "")[:10]
        source = r.get("source", "")
        text = (r.get("text") or "")[:500]
        context_parts.append(f"[Memory {i + 1}] ({date}, {source}) {text}")

    prompt = _REFLECT_PROMPT.format(
        context="\n\n".join(context_parts),
        query=query,
    )

    # -- 3. LLM call --------------------------------------------------------
    answer = _call_llm(prompt)

    # -- 4. Return ----------------------------------------------------------
    sources = [
        {
            "point_id": r.get("point_id"),
            "text": (r.get("text") or "")[:200],
            "score": r.get("final_score") or r.get("score", 0),
        }
        for r in results[:15]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "search_elapsed_ms": search_ms,
        "total_elapsed_ms": round((time.time() - start) * 1000, 1),
        "reflect_model": REFLECT_MODEL,
    }


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


def _call_llm(prompt: str) -> str:
    """Route to the configured LLM provider."""
    if REFLECT_PROVIDER == "anthropic" and _state.ANTHROPIC_API_KEY:
        return _call_anthropic(prompt)
    if REFLECT_PROVIDER == "ollama":
        return _call_ollama(prompt)
    # Fallback: Anthropic if key exists, else Ollama
    if _state.ANTHROPIC_API_KEY:
        return _call_anthropic(prompt)
    return _call_ollama(prompt)


def _call_anthropic(prompt: str) -> str:
    body = json.dumps(
        {
            "model": REFLECT_MODEL,
            "max_tokens": REFLECT_MAX_TOKENS,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", _state.ANTHROPIC_API_KEY)
    req.add_header("anthropic-version", "2023-06-01")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["content"][0]["text"].strip()
    except Exception as exc:
        _state.logger.error("Reflect Anthropic call failed: %s", exc)
        return f"Reflection failed: {exc}"


def _call_ollama(prompt: str) -> str:
    ollama_url = os.environ.get("REFLECT_OLLAMA_URL", _state.AMAC_LLM_URL)
    ollama_model = os.environ.get("REFLECT_OLLAMA_MODEL", "qwen2.5:14b")
    body = json.dumps(
        {
            "model": ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": REFLECT_MAX_TOKENS,
        }
    ).encode()
    req = urllib.request.Request(ollama_url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        _state.logger.error("Reflect Ollama call failed: %s", exc)
        return f"Reflection failed: {exc}"
