#!/usr/bin/env python3

import importlib
import sys
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

safe_import = importlib.import_module("pipeline._imports").safe_import

_qdrant_batch = safe_import("pipeline.qdrant_batch", "tools.pipeline.qdrant_batch")
scroll_all = _qdrant_batch.scroll_all

_config_module = safe_import("config", "tools.config")
load_config = _config_module.load_config


def backfill_schema(batch_size: int = 100) -> int:
    cfg = load_config()
    qdrant = QdrantClient(url=cfg["qdrant"]["url"])
    collection = cfg["qdrant"]["collection"]

    total = 0
    embedding_model = cfg["embeddings"]["model"]
    ids_batch: list[Any] = []

    for point in scroll_all(
        qdrant_client=qdrant,
        collection=collection,
        batch_size=batch_size,
        with_payload=False,
        with_vectors=False,
    ):
        ids_batch.append(point.id)
        if len(ids_batch) < batch_size:
            continue
        qdrant.set_payload(
            collection_name=collection,
            points=PointIdsList(points=cast(list[Any], ids_batch)),
            payload={"embedding_model": embedding_model, "schema_version": "0.3"},
        )
        total += len(ids_batch)
        print(f"Backfilled {total} points", flush=True)
        ids_batch = []

    if ids_batch:
        qdrant.set_payload(
            collection_name=collection,
            points=PointIdsList(points=cast(list[Any], ids_batch)),
            payload={"embedding_model": embedding_model, "schema_version": "0.3"},
        )
        total += len(ids_batch)
        print(f"Backfilled {total} points", flush=True)

    return total


if __name__ == "__main__":
    updated = backfill_schema()
    print(f"Completed schema backfill for {updated} points", flush=True)
    sys.exit(0)
