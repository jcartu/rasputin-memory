from __future__ import annotations

from datetime import datetime, timezone

from brain.schema import Entity, EntityRef, EntityUnitJoin, MemoryLink, MemoryUnit, model_dump_compat


def test_memory_unit_round_trip_normalizes_legacy_fact_type() -> None:
    unit = MemoryUnit(
        id="550e8400-e29b-41d4-a716-446655440000",
        bank_id="conv_26",
        text="Alice changed jobs | Involving: Alice | Date: 2024-01-01",
        fact_type="inference",
        mentioned_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        entities=[EntityRef(name="Alice", type="Person", role="subject")],
        tags=["fact"],
        metadata={"source": "migration"},
    )

    dumped = model_dump_compat(unit)
    restored = MemoryUnit(**dumped)

    assert restored.fact_type == "observation"
    assert restored.entities[0].name == "Alice"
    assert restored.metadata["source"] == "migration"


def test_memory_link_round_trip() -> None:
    link = MemoryLink(
        id="550e8400-e29b-41d4-a716-446655440001",
        bank_id="conv_26",
        from_unit_id="u1",
        to_unit_id="u2",
        link_type="semantic",
        weight=0.7,
        metadata={"reason": "close paraphrase"},
    )
    assert MemoryLink(**model_dump_compat(link)) == link


def test_entity_round_trip_normalizes_type() -> None:
    entity = Entity(
        id="550e8400-e29b-41d4-a716-446655440002",
        bank_id="conv_26",
        canonical_name="Alice",
        aliases=["Alice Smith"],
        entity_type="Person",
        mention_count=3,
    )

    restored = Entity(**model_dump_compat(entity))

    assert restored.entity_type == "PERSON"
    assert restored.aliases == ["Alice Smith"]


def test_entity_ref_round_trip() -> None:
    entity_ref = EntityRef(name="Paris", type="Location", role="location")
    assert EntityRef(**model_dump_compat(entity_ref)) == entity_ref


def test_entity_unit_join_round_trip() -> None:
    join = EntityUnitJoin(entity_id="e1", unit_id="u1", role="subject", bank_id="conv_26")
    assert EntityUnitJoin(**model_dump_compat(join)) == join
