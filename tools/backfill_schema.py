#!/usr/bin/env python3

import importlib
import sys
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

try:
    _config_module = importlib.import_module("config")
except ModuleNotFoundError:
    _config_module = importlib.import_module("tools.config")
load_config = _config_module.load_config


def backfill_schema(batch_size: int = 100) -> int:
    cfg = load_config()
    qdrant = QdrantClient(url=cfg["qdrant"]["url"])
    collection = cfg["qdrant"]["collection"]

    offset = None
    total = 0
    embedding_model = cfg["embeddings"]["model"]

    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            offset=offset,
            limit=batch_size,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break

        ids = [point.id for point in points]
        qdrant.set_payload(
            collection_name=collection,
            points=PointIdsList(points=cast(list[Any], ids)),
            payload={"embedding_model": embedding_model, "schema_version": "3.0"},
        )
        total += len(ids)
        print(f"Backfilled {total} points", flush=True)

    return total


if __name__ == "__main__":
    updated = backfill_schema()
    print(f"Completed schema backfill for {updated} points", flush=True)
    sys.exit(0)
