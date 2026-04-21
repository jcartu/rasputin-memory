from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest

from brain.graph_store import GraphStore
from brain.schema import EntityRef, MemoryLink, MemoryUnit
from brain.sqlite_store import SqliteStore


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, dict[str, object]]] = {}

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise RuntimeError("missing collection")
        points_count = len(self.collections[collection_name])
        return SimpleNamespace(points_count=points_count)

    def create_collection(self, collection_name: str, vectors_config=None):
        self.collections.setdefault(collection_name, {})

    def create_payload_index(self, collection_name: str, field_name: str, field_schema=None):
        self.collections.setdefault(collection_name, {})

    def upsert(self, collection_name: str, points):
        bucket = self.collections.setdefault(collection_name, {})
        for point in points:
            bucket[str(point.id)] = {"payload": dict(point.payload), "vector": list(point.vector)}

    def retrieve(self, collection_name: str, ids, with_payload=True, with_vectors=False):
        bucket = self.collections.get(collection_name, {})
        results = []
        for point_id in ids:
            if str(point_id) not in bucket:
                continue
            record = bucket[str(point_id)]
            results.append(
                SimpleNamespace(
                    id=str(point_id),
                    payload=record["payload"] if with_payload else None,
                    vector=record["vector"] if with_vectors else None,
                )
            )
        return results


def test_graph_store_end_to_end_add_unit_resolve_entity_expand_links(tmp_path, monkeypatch) -> None:
    fake_qdrant = FakeQdrant()
    sqlite_store = SqliteStore(str(tmp_path / "graph.db"))
    graph_store = GraphStore(
        qdrant_url="http://localhost:6333",
        sqlite_path=str(tmp_path / "graph.db"),
        embed_dim=4,
        qdrant_client=fake_qdrant,
        sqlite_store=sqlite_store,
    )
    monkeypatch.setattr("brain.embedding.get_embedding", lambda *args, **kwargs: [0.1, 0.2, 0.3, 0.4])

    unit = MemoryUnit(
        id=str(uuid.uuid4()),
        bank_id="bank-a",
        text="Alice moved to Toronto",
        fact_type="world",
        entities=[EntityRef(name="Alice", type="Person", role="subject")],
        tags=["fact"],
    )
    graph_store.add_unit(unit)
    graph_store.add_link(
        MemoryLink(
            id=str(uuid.uuid4()),
            bank_id="bank-a",
            from_unit_id=unit.id,
            to_unit_id="neighbor-1",
            link_type="semantic",
            weight=0.8,
        )
    )

    loaded = graph_store.get_unit(unit.id, "bank-a")
    resolved = graph_store.resolve_entity("Alic", "bank-a")
    links = graph_store.expand_links([unit.id], "bank-a", ["semantic"])

    assert loaded is not None
    assert loaded.text == unit.text
    assert resolved is not None
    assert resolved.canonical_name == "Alice"
    assert [link.to_unit_id for link in links] == ["neighbor-1"]


@pytest.mark.perf
def test_perf_batch_add_units_under_5_seconds(tmp_path, monkeypatch) -> None:
    fake_qdrant = FakeQdrant()
    sqlite_store = SqliteStore(str(tmp_path / "graph.db"))
    graph_store = GraphStore(
        qdrant_url="http://localhost:6333",
        sqlite_path=str(tmp_path / "graph.db"),
        embed_dim=8,
        qdrant_client=fake_qdrant,
        sqlite_store=sqlite_store,
    )
    monkeypatch.setattr("brain.embedding.get_embedding", lambda *args, **kwargs: [0.1] * 8)
    units = [
        MemoryUnit(
            id=str(uuid.uuid4()),
            bank_id="bank-a",
            text=f"Fact {idx}",
            fact_type="world",
            entities=[EntityRef(name=f"Entity {idx}", type="Thing")],
        )
        for idx in range(1000)
    ]

    start = time.perf_counter()
    graph_store.batch_add_units(units)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0
    assert fake_qdrant.get_collection("memory_units_bank-a").points_count == 1000
