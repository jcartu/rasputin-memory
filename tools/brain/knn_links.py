from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
KNN_TOP_K = int(os.environ.get("KNN_TOP_K", "30"))
KNN_THRESHOLD = float(os.environ.get("KNN_THRESHOLD", "0.6"))
KNN_EXPAND_MAX = int(os.environ.get("KNN_EXPAND_MAX", "10"))


def _http_json(url: str, data: dict[str, Any] | None = None, method: str | None = None, timeout: int = 10) -> Any:
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def compute_links_for_point(
    collection: str,
    point_id: int | str,
    vector: list[float],
    top_k: int = KNN_TOP_K,
    threshold: float = KNN_THRESHOLD,
) -> list[int | str]:
    """Search Qdrant for top-K most similar existing points with score >= threshold.

    Returns list of similar point IDs, excluding self.
    """
    data = _http_json(
        f"{QDRANT_URL}/collections/{collection}/points/query",
        data={
            "query": vector,
            "limit": top_k + 1,
            "with_payload": False,
            "score_threshold": threshold,
        },
        method="POST",
    )
    similar_ids: list[int | str] = []
    for p in data.get("result", {}).get("points", []):
        pid = p["id"]
        if pid != point_id and p.get("score", 0) >= threshold:
            similar_ids.append(pid)
    return similar_ids[:top_k]


def store_links(collection: str, point_id: int | str, similar_ids: list[int | str]) -> None:
    """Store similar_ids as a payload field on the given point."""
    _http_json(
        f"{QDRANT_URL}/collections/{collection}/points/payload",
        data={
            "points": [point_id],
            "payload": {"similar_ids": similar_ids},
        },
        method="POST",
    )


def expand_seeds(
    collection: str,
    seed_ids: list[int | str],
    exclude_ids: set[int | str],
) -> list[dict[str, Any]]:
    """Batch-fetch similar_ids from seed payloads, then fetch the linked facts.

    Returns formatted result dicts with origin='knn_expansion' and score=0.5
    (neutral — CE will re-rank).
    """
    if not seed_ids:
        return []

    linked_ids: set[int | str] = set()
    try:
        data = _http_json(
            f"{QDRANT_URL}/collections/{collection}/points",
            data={
                "ids": list(seed_ids),
                "with_payload": {"include": ["similar_ids"]},
            },
            method="POST",
            timeout=15,
        )
        for point in data.get("result", []):
            payload = point.get("payload", {})
            for sid in payload.get("similar_ids", []):
                if sid not in exclude_ids and sid not in seed_ids:
                    linked_ids.add(sid)
    except Exception as exc:
        logger.warning("kNN expand_seeds payload fetch failed: %s", exc)
        return []

    if not linked_ids:
        return []

    results: list[dict[str, Any]] = []
    try:
        data = _http_json(
            f"{QDRANT_URL}/collections/{collection}/points",
            data={
                "ids": list(linked_ids),
                "with_payload": True,
            },
            method="POST",
            timeout=15,
        )
        for point in data.get("result", []):
            payload = point.get("payload", {})
            results.append(
                {
                    "score": 0.5,
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "date": payload.get("date", ""),
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "domain": payload.get("domain", ""),
                    "importance": payload.get("importance", 50),
                    "retrieval_count": payload.get("retrieval_count", 0),
                    "last_accessed": payload.get("last_accessed", ""),
                    "point_id": point["id"],
                    "chunk_type": payload.get("chunk_type", ""),
                    "fact_type": payload.get("fact_type", ""),
                    "origin": "knn_expansion",
                }
            )
    except Exception as exc:
        logger.warning("kNN expand_seeds fact fetch failed: %s", exc)

    return results[:KNN_EXPAND_MAX]
