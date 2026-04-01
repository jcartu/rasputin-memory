#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import math
import random
from typing import Any, Callable

import requests
from qdrant_client import QdrantClient

safe_import = importlib.import_module("pipeline._imports").safe_import

_config_module = safe_import("config", "tools.config")
load_config = _config_module.load_config


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def check_embedding_consistency(
    qdrant_client: Any,
    collection: str,
    embed_fn: Callable[[str], list[float]],
    sample_size: int = 100,
    threshold: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    limit = max(sample_size * 3, sample_size)
    points, _ = qdrant_client.scroll(
        collection_name=collection,
        limit=limit,
        with_vectors=True,
        with_payload=True,
    )

    candidates = [p for p in (points or []) if (p.payload or {}).get("text") and getattr(p, "vector", None)]
    if not candidates:
        return {"total": 0, "drifted": 0, "drift_rate": 0.0, "threshold": threshold}

    rng = random.Random(seed)
    if len(candidates) > sample_size:
        sample = rng.sample(candidates, sample_size)
    else:
        sample = candidates

    drifted = 0
    for point in sample:
        text = (point.payload or {}).get("text", "")
        current_embedding = embed_fn(text)
        stored_vector = point.vector
        if isinstance(stored_vector, dict):
            stored_vector = next(iter(stored_vector.values()))
        sim = cosine_similarity(current_embedding, stored_vector)
        if sim < threshold:
            drifted += 1

    total = len(sample)
    return {
        "total": total,
        "drifted": drifted,
        "drift_rate": (drifted / total) if total else 0.0,
        "threshold": threshold,
    }


def build_embed_fn(embed_url: str, embed_model: str, prefix: str) -> Callable[[str], list[float]]:
    def _embed(text: str) -> list[float]:
        resp = requests.post(
            embed_url,
            json={"model": embed_model, "input": f"{prefix}{text}"},
            timeout=35,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data:
            return data["embeddings"][0]
        return data["embedding"]

    return _embed


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Embedding drift health check")
    parser.add_argument("--qdrant-url", default=cfg["qdrant"]["url"])
    parser.add_argument("--collection", default=cfg["qdrant"]["collection"])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    embed_fn = build_embed_fn(
        embed_url=cfg["embeddings"]["url"],
        embed_model=cfg["embeddings"]["model"],
        prefix=cfg["embeddings"]["prefix_doc"],
    )
    result = check_embedding_consistency(
        qdrant_client=client,
        collection=args.collection,
        embed_fn=embed_fn,
        sample_size=max(1, args.sample_size),
        threshold=args.threshold,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
