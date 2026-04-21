#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from qdrant_client import QdrantClient  # noqa: E402  (post sys.path insert)

from brain.graph_store import GraphStore  # noqa: E402
from brain.schema import EntityRef, MemoryUnit, get_configured_embed_dim, normalize_fact_type  # noqa: E402
from config import load_config  # noqa: E402
from pipeline.qdrant_batch import scroll_all  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate v0.9.1 LoCoMo facts into memory_units collections")
    parser.add_argument("--source-collection", required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for fmt in (
        None,
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            if fmt is None:
                parsed = datetime.fromisoformat(text)
            else:
                parsed = datetime.strptime(text, fmt)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_entities(raw_value: Any) -> list[EntityRef]:
    if raw_value in (None, "", []):
        return []
    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)
    else:
        parsed = raw_value
    entities: list[EntityRef] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        entities.append(
            EntityRef(
                name=str(item.get("name", "")).strip(),
                type=str(item.get("type", "OTHER")).strip() or "OTHER",
                role=item.get("role"),
            )
        )
    return [entity for entity in entities if entity.name]


def _point_to_unit(point: Any, source_collection: str, bank_id: str) -> tuple[MemoryUnit, list[float]]:
    payload = dict(point.payload or {})
    entities = _parse_entities(payload.get("entities"))
    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "text",
            "fact_type",
            "occurred_start",
            "occurred_end",
            "entities",
            "date",
            "where",
            "_ingest_commit_sha",
            "_ingest_config_hash",
        }
    }
    unit = MemoryUnit(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_collection}:{point.id}")),
        bank_id=bank_id,
        text=str(payload.get("text", "")),
        context=source_collection,
        fact_type=normalize_fact_type(str(payload.get("fact_type", "world"))),
        event_date=_parse_datetime(payload.get("event_date")),
        occurred_start=_parse_datetime(payload.get("occurred_start")),
        occurred_end=_parse_datetime(payload.get("occurred_end")),
        mentioned_at=_parse_datetime(payload.get("date")),
        where=payload.get("where"),
        entities=entities,
        tags=[str(payload.get("chunk_type"))] if payload.get("chunk_type") else [],
        proof_count=1,
        metadata=metadata,
        _ingest_commit_sha=payload.get("_ingest_commit_sha"),
        _ingest_config_hash=payload.get("_ingest_config_hash"),
    )
    vector = list(point.vector or [])
    return unit, vector


def migrate_collection(
    *,
    qdrant_client: QdrantClient,
    graph_store: GraphStore,
    source_collection: str,
    bank_id: str,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    target_collection = f"memory_units_{bank_id}"
    source_count = int(qdrant_client.get_collection(source_collection).points_count or 0)
    expected_count = min(source_count, limit) if limit else source_count

    if not dry_run and not force:
        try:
            target_count = int(qdrant_client.get_collection(target_collection).points_count or 0)
            if target_count == expected_count:
                return {
                    "already_migrated": True,
                    "message": f"{target_collection} already migrated ({target_count} units)",
                    "units_migrated": 0,
                    "entities_created": 0,
                    "entity_units_created": 0,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
        except Exception:
            pass

    if not dry_run and force:
        try:
            qdrant_client.delete_collection(target_collection)
        except Exception:
            pass
        graph_store.sqlite.delete_bank(bank_id)

    units: list[MemoryUnit] = []
    vectors: list[list[float]] = []
    unique_entities: set[tuple[str, str]] = set()
    entity_unit_pairs = 0

    for index, point in enumerate(scroll_all(qdrant_client, source_collection, batch_size=128, with_payload=True, with_vectors=True)):
        if limit is not None and index >= limit:
            break
        unit, vector = _point_to_unit(point, source_collection, bank_id)
        units.append(unit)
        vectors.append(vector)
        for entity in unit.entities:
            unique_entities.add((entity.name, entity.type))
            entity_unit_pairs += 1

    if not dry_run:
        graph_store.upsert_units_with_vectors(units, vectors)

    return {
        "already_migrated": False,
        "message": f"migrated {len(units)} units from {source_collection} to {target_collection}",
        "units_migrated": len(units),
        "entities_created": len(unique_entities) if dry_run else graph_store.sqlite.count_rows("entities", bank_id=bank_id),
        "entity_units_created": entity_unit_pairs if dry_run else graph_store.sqlite.count_rows("entity_units", bank_id=bank_id),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    args = _parse_args()
    config = load_config("config/rasputin.toml")
    qdrant_client = QdrantClient(url=config["qdrant"]["url"])
    graph_store = GraphStore(
        qdrant_url=config["qdrant"]["url"],
        sqlite_path=config["graph_store"]["sqlite_path"],
        embed_dim=get_configured_embed_dim(),
        qdrant_client=qdrant_client,
    )
    graph_store.init_schema()

    result = migrate_collection(
        qdrant_client=qdrant_client,
        graph_store=graph_store,
        source_collection=args.source_collection,
        bank_id=args.bank_id,
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
