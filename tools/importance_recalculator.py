#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timedelta, timezone

from qdrant_client import QdrantClient

from pipeline.dateparse import parse_date
from pipeline.scoring_constants import get_source_weight

safe_import = importlib.import_module("pipeline._imports").safe_import
scroll_all = safe_import("pipeline.qdrant_batch", "tools.pipeline.qdrant_batch").scroll_all
load_config = safe_import("config", "tools.config").load_config


def recalculate_importance(
    qdrant_client,
    collection: str,
    execute: bool = False,
    batch_size: int = 256,
    hot_topics: set[str] | None = None,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    topics = hot_topics or set()
    scanned = 0
    updated = 0
    for point in scroll_all(
        qdrant_client=qdrant_client, collection=collection, batch_size=batch_size, with_payload=True
    ):
        scanned += 1
        payload = point.payload or {}
        base = int(payload.get("importance", 50) or 50)
        recency = parse_date(payload.get("last_accessed")) or parse_date(payload.get("date"))
        source_bonus = int((get_source_weight(str(payload.get("source", ""))) - 0.5) * 10)
        retrieval_bonus = 10 if int(payload.get("retrieval_count", 0) or 0) > 5 else 0
        stale_penalty = 10 if recency and (now - recency) > timedelta(days=90) else 0
        text_blob = f"{payload.get('text', '')} {payload.get('source', '')}".lower()
        topic_bonus = 5 if topics and any(topic.lower() in text_blob for topic in topics) else 0
        score = max(0, min(100, base + source_bonus + retrieval_bonus + topic_bonus - stale_penalty))
        if score == base:
            continue
        updated += 1
        if execute:
            qdrant_client.set_payload(
                collection_name=collection,
                points=[point.id],
                payload={"importance": score, "importance_recalculated_at": now.isoformat()},
            )
    return {"ok": True, "execute": execute, "scanned": scanned, "updated": updated}


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Recalculate memory importance scores")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--qdrant-url", default=cfg["qdrant"]["url"])
    parser.add_argument("--collection", default=cfg["qdrant"]["collection"])
    args = parser.parse_args()
    print(
        recalculate_importance(
            qdrant_client=QdrantClient(url=args.qdrant_url),
            collection=args.collection,
            execute=args.execute,
            batch_size=max(1, args.batch_size),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
