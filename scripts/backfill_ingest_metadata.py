#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from brain.ingest_metadata import get_ingest_metadata  # noqa: E402  (post sys.path insert)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill ingest metadata onto an existing Qdrant collection.")
    parser.add_argument("--collection", required=True, help="Qdrant collection to update")
    parser.add_argument("--commit", required=True, help="40-char git SHA to stamp into _ingest_commit_sha")
    parser.add_argument("--config-hash", default=None, help="Override _ingest_config_hash")
    parser.add_argument("--timestamp", default=None, help="Override _ingest_timestamp")
    parser.add_argument("--version", default=None, help="Override _ingest_bench_version")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant base URL")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    return parser.parse_args(argv)


def build_metadata(args: argparse.Namespace) -> dict[str, str]:
    defaults = get_ingest_metadata()
    return {
        "_ingest_commit_sha": args.commit,
        "_ingest_config_hash": args.config_hash or defaults["_ingest_config_hash"],
        "_ingest_timestamp": args.timestamp or defaults["_ingest_timestamp"],
        "_ingest_bench_version": args.version or defaults["_ingest_bench_version"],
    }


def backfill_collection(
    client: QdrantClient,
    collection: str,
    metadata: dict[str, str],
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    scanned = 0
    updated = 0
    offset: Any = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        pending_ids: list[Any] = []
        for point in points:
            scanned += 1
            payload = point.payload or {}
            if payload.get("_ingest_commit_sha"):
                continue
            pending_ids.append(point.id)

        if pending_ids:
            updated += len(pending_ids)
            if dry_run:
                print(f"DRY RUN {collection}: would update {len(pending_ids)} point(s): {pending_ids[:10]}")
            else:
                client.set_payload(collection_name=collection, points=pending_ids, payload=metadata)
                print(f"UPDATED {collection}: {len(pending_ids)} point(s)")

        if offset is None:
            break

    return scanned, updated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_metadata(args)
    client = QdrantClient(url=args.qdrant_url)
    scanned, updated = backfill_collection(client, args.collection, metadata, dry_run=args.dry_run)
    verb = "would update" if args.dry_run else "updated"
    print(f"DONE {args.collection}: scanned={scanned} {verb}={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
