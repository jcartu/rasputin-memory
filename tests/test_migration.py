from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from brain.graph_store import GraphStore
from brain.schema import get_configured_embed_dim
from scripts.migrate_v091_to_memory_units import migrate_collection


@pytest.mark.skipif(not os.environ.get("RUN_MIGRATION_INTEGRATION"), reason="integration: requires live Qdrant locomo_lb_conv_26 with 768d vectors; set RUN_MIGRATION_INTEGRATION=1")
def test_migration_conv26_parity(tmp_path) -> None:
    qdrant = QdrantClient(url="http://localhost:6333")
    source_collection = "locomo_lb_conv_26"
    source_count = int(qdrant.get_collection(source_collection).points_count or 0)
    assert source_count == 1311

    bank_id = f"test_conv_26_{uuid.uuid4().hex[:8]}"
    graph_store = GraphStore(
        qdrant_url="http://localhost:6333",
        sqlite_path=str(tmp_path / "graph.db"),
        embed_dim=get_configured_embed_dim(),
        qdrant_client=qdrant,
    )
    result = migrate_collection(
        qdrant_client=qdrant,
        graph_store=graph_store,
        source_collection=source_collection,
        bank_id=bank_id,
    )

    target_collection = f"memory_units_{bank_id}"
    target_count = int(qdrant.get_collection(target_collection).points_count or 0)

    source_points, _ = qdrant.scroll(collection_name=source_collection, limit=source_count, with_payload=True, with_vectors=False)
    source_entities = set()
    for point in source_points:
        raw_entities = point.payload.get("entities") if point.payload else None
        if not raw_entities:
            continue
        import json

        for item in json.loads(raw_entities):
            source_entities.add(item["name"])

    resolved_rows = graph_store.sqlite.count_rows("entities", bank_id=bank_id)

    assert result["units_migrated"] == source_count
    assert target_count == source_count
    assert resolved_rows == len(source_entities)
