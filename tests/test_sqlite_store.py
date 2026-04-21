from __future__ import annotations

import statistics
import threading
import time
import uuid
from datetime import datetime, timezone

import pytest

from brain.schema import Entity, EntityUnitJoin, MemoryLink
from brain.sqlite_store import SqliteStore


def _store(tmp_path, threshold: float = 0.4) -> SqliteStore:
    store = SqliteStore(str(tmp_path / "graph.db"), entity_match_threshold=threshold)
    store.init_schema()
    return store


def test_sqlite_store_crud_and_fts(tmp_path) -> None:
    store = _store(tmp_path)
    entity = Entity(
        id=str(uuid.uuid4()),
        bank_id="bank-a",
        canonical_name="Caroline Alvarez",
        aliases=["Caroline", "Caro"],
        entity_type="PERSON",
        first_mentioned_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_mentioned_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        mention_count=1,
    )
    store.upsert_entity(entity)
    resolved = store.resolve_entity("Carolin Alvarez", "bank-a")

    assert resolved is not None
    assert resolved.canonical_name == "Caroline Alvarez"

    link = MemoryLink(
        id=str(uuid.uuid4()),
        bank_id="bank-a",
        from_unit_id="u1",
        to_unit_id="u2",
        link_type="semantic",
        weight=0.9,
    )
    store.add_link(link)
    links = store.expand_links(["u1"], "bank-a", ["semantic"])

    assert [item.to_unit_id for item in links] == ["u2"]

    join_count = store.batch_add_entity_units([EntityUnitJoin(entity_id=entity.id, unit_id="u1", role="subject", bank_id="bank-a")])
    assert join_count == 1


def test_explain_query_plan_uses_forward_index(tmp_path) -> None:
    store = _store(tmp_path)
    for idx in range(5):
        store.add_link(
            MemoryLink(
                id=str(uuid.uuid4()),
                bank_id="bank-a",
                from_unit_id="u1",
                to_unit_id=f"u{idx}",
                link_type="semantic",
                weight=0.9 - (idx * 0.1),
            )
        )
    plan = store.explain_expand_links_query("bank-a", "u1", ["semantic"])
    assert any("idx_links_forward" in line for line in plan)


def test_transactional_correctness_under_concurrent_reads(tmp_path) -> None:
    store = _store(tmp_path)
    entity = Entity(id=str(uuid.uuid4()), bank_id="bank-a", canonical_name="Alice", aliases=[], entity_type="PERSON")
    store.upsert_entity(entity)

    barrier = threading.Barrier(5)
    failures: list[str] = []

    def reader() -> None:
        barrier.wait()
        for _ in range(50):
            resolved = store.resolve_entity("Alice", "bank-a")
            if resolved is None or resolved.canonical_name != "Alice":
                failures.append("missing entity")

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()

    for idx in range(10):
        store.add_link(
            MemoryLink(
                id=str(uuid.uuid4()),
                bank_id="bank-a",
                from_unit_id="u1",
                to_unit_id=f"u{idx}",
                link_type="temporal",
                weight=0.5,
            )
        )

    for thread in threads:
        thread.join()

    assert failures == []
    assert len(store.expand_links(["u1"], "bank-a", ["temporal"])) == 10


@pytest.mark.perf
def test_perf_resolve_entity_p95_under_5ms(tmp_path) -> None:
    store = _store(tmp_path, threshold=0.3)
    for idx in range(300):
        store.upsert_entity(
            Entity(
                id=str(uuid.uuid4()),
                bank_id="bank-a",
                canonical_name=f"Entity {idx}",
                aliases=[f"Alias {idx}"],
                entity_type="THING",
                mention_count=idx + 1,
            )
        )
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        resolved = store.resolve_entity("Entit 299", "bank-a")
        timings.append((time.perf_counter() - start) * 1000)
        assert resolved is not None
    assert statistics.quantiles(timings, n=100)[94] < 5.0


@pytest.mark.perf
def test_perf_expand_links_p95_under_10ms_per_source(tmp_path) -> None:
    store = _store(tmp_path)
    for source_idx in range(10):
        for target_idx in range(100):
            store.add_link(
                MemoryLink(
                    id=str(uuid.uuid4()),
                    bank_id="bank-a",
                    from_unit_id=f"u{source_idx}",
                    to_unit_id=f"t{source_idx}-{target_idx}",
                    link_type="semantic",
                    weight=1.0 - (target_idx / 1000),
                )
            )
    timings = []
    for source_idx in range(10):
        start = time.perf_counter()
        links = store.expand_links([f"u{source_idx}"], "bank-a", ["semantic"], limit_per_source=10)
        timings.append((time.perf_counter() - start) * 1000)
        assert len(links) == 10
    assert statistics.quantiles(timings, n=100)[94] < 10.0
