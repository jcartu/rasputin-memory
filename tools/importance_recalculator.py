#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any

from qdrant_client import QdrantClient

from pipeline.dateparse import parse_date

try:
    _qdrant_batch = importlib.import_module("pipeline.qdrant_batch")
except ModuleNotFoundError:
    _qdrant_batch = importlib.import_module("tools.pipeline.qdrant_batch")
scroll_all = _qdrant_batch.scroll_all

try:
    _config_module = importlib.import_module("config")
except ModuleNotFoundError:
    _config_module = importlib.import_module("tools.config")
load_config = _config_module.load_config


def get_recent_commit_topics(limit: int = 30) -> set[str]:
    try:
        out = subprocess.check_output(["git", "log", f"-{limit}", "--pretty=%s"], text=True)
    except Exception:
        return set()
    tokens: set[str] = set()
    for line in out.splitlines():
        for raw in line.lower().replace("(", " ").replace(")", " ").replace(":", " ").split():
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_"})
            if len(cleaned) >= 4:
                tokens.add(cleaned)
    return tokens


def _is_topic_hot(payload: dict[str, Any], hot_topics: set[str]) -> bool:
    if not hot_topics:
        return False
    text = str(payload.get("text", "")).lower()
    source = str(payload.get("source", "")).lower()
    tags = payload.get("tags", [])
    if isinstance(tags, list):
        tag_text = " ".join(str(t).lower() for t in tags)
    else:
        tag_text = str(tags).lower()
    blob = f"{text} {source} {tag_text}"
    return any(topic in blob for topic in hot_topics)


def calculate_new_importance(payload: dict[str, Any], now: datetime, hot_topics: set[str]) -> int:
    current = payload.get("importance", 50)
    try:
        score = int(current)
    except (TypeError, ValueError):
        score = 50

    retrieval_count = payload.get("retrieval_count", 0) or 0
    try:
        retrieval_count = int(retrieval_count)
    except (TypeError, ValueError):
        retrieval_count = 0

    if retrieval_count > 5:
        score += 10

    last_accessed = parse_date(payload.get("last_accessed")) or parse_date(payload.get("date"))
    if last_accessed and (now - last_accessed) > timedelta(days=90):
        score -= 10

    if _is_topic_hot(payload, hot_topics):
        score += 5

    return max(0, min(100, score))


def recalculate_importance(
    qdrant_client: Any,
    collection: str,
    execute: bool = False,
    batch_size: int = 256,
    hot_topics: set[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    topics = hot_topics if hot_topics is not None else get_recent_commit_topics()

    updated = 0
    scanned = 0
    for point in scroll_all(
        qdrant_client=qdrant_client,
        collection=collection,
        batch_size=batch_size,
        with_payload=True,
        with_vectors=False,
    ):
        scanned += 1
        payload = point.payload or {}
        new_importance = calculate_new_importance(payload, now, topics)
        old_importance = payload.get("importance", 50)
        try:
            old_importance = int(old_importance)
        except (TypeError, ValueError):
            old_importance = 50

        if new_importance != old_importance:
            updated += 1
            if execute:
                qdrant_client.set_payload(
                    collection_name=collection,
                    points=[point.id],
                    payload={"importance": new_importance, "importance_recalculated_at": now.isoformat()},
                )

    return {
        "ok": True,
        "execute": execute,
        "scanned": scanned,
        "updated": updated,
        "hot_topics": len(topics),
    }


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Recalculate memory importance scores")
    parser.add_argument("--execute", action="store_true", help="Apply updates (default is dry-run)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--qdrant-url", default=cfg["qdrant"]["url"])
    parser.add_argument("--collection", default=cfg["qdrant"]["collection"])
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    result = recalculate_importance(
        qdrant_client=client,
        collection=args.collection,
        execute=args.execute,
        batch_size=max(1, args.batch_size),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
