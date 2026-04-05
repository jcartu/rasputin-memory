from __future__ import annotations

import math
import os as _os
import re
import time
from typing import Any, Optional

from brain import _state

_TARGET_DIMS = int(_os.environ.get("EMBED_DIM", _state.CONFIG.get("embeddings", {}).get("dimensions", 768)))


def is_reranker_available() -> bool:
    if not _state.RERANKER_ENABLED:
        return False
    try:
        response = _state.requests.post(
            _state.RERANKER_URL,
            json={"query": "ping", "passages": ["ping"]},
            timeout=2,
        )
        return response.status_code == 200
    except Exception:
        return False


def get_embedding(text: str, prefix: str = _state.EMBED_PREFIX_QUERY) -> list[float]:
    try:
        from brain.embedding_providers import EMBED_PROVIDER, get_embedding_auto

        if EMBED_PROVIDER != "ollama":
            return get_embedding_auto(text, prefix=prefix)
    except ImportError:
        pass

    return _get_embedding_ollama(text, prefix=prefix)


def _truncate_if_needed(vec: list[float]) -> list[float]:
    if len(vec) <= _TARGET_DIMS:
        return vec
    truncated = vec[:_TARGET_DIMS]
    mag = math.sqrt(sum(v * v for v in truncated))
    if mag > 0:
        return [v / mag for v in truncated]
    return truncated


def _get_embedding_ollama(text: str, prefix: str = _state.EMBED_PREFIX_QUERY) -> list[float]:
    prefixed_text = f"{prefix}{text}" if prefix else text
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = _state.requests.post(
                _state.EMBED_URL,
                json={"model": _state.EMBED_MODEL, "input": prefixed_text},
                timeout=35,
            )
            response.raise_for_status()
            data = response.json()
            if "embeddings" in data:
                return _truncate_if_needed(data["embeddings"][0])
            if "data" in data and isinstance(data["data"], list):
                return _truncate_if_needed(data["data"][0]["embedding"])
            if "embedding" in data:
                return _truncate_if_needed(data["embedding"])
            raise ValueError(f"Unexpected embedding response: {list(data.keys())}")
        except _state.requests.exceptions.Timeout:
            _state.logger.warning("Embedding timeout on attempt %s/%s", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception("Embedding service timeout after retries")
        except _state.requests.exceptions.ConnectionError as error:
            _state.logger.error("Embedding connection error: %s", error)
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception(f"Embedding service unavailable: {error}")
        except Exception as error:
            _state.logger.error("Embedding error: %s", error)
            raise

    raise RuntimeError("Embedding generation failed")


def check_duplicate(vector: list[float], text: str, threshold: float = 0.92) -> tuple[bool, Optional[Any], float]:
    try:
        results = _state.qdrant.query_points(  # type: ignore[attr-defined]  # qdrant-client>=1.9.0
            collection_name=_state.COLLECTION,
            query=vector,
            limit=3,
            with_payload=True,
        )
        for point in results.points:
            if point.score >= threshold:
                payload = point.payload or {}
                existing_text = payload.get("text", "")
                words_new = set(re.findall(r"\w+", text.lower()))
                words_old = set(re.findall(r"\w+", existing_text.lower()))
                overlap = len(words_new & words_old) / max(len(words_new | words_old), 1)
                if overlap > 0.5 or point.score >= 0.95:
                    return True, point.id, round(point.score, 4)
        return False, None, 0
    except Exception as error:
        _state.logger.error("Dedup check error: %s", error)
        return False, None, 0
