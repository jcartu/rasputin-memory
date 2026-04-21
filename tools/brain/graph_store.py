from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from brain import _state, embedding
from brain.schema import Entity, EntityUnitJoin, MemoryLink, MemoryUnit, model_dump_compat, model_validate_compat
from brain.sqlite_store import SqliteStore


class GraphStore:
    def __init__(
        self,
        *,
        qdrant_url: str,
        sqlite_path: str,
        embed_dim: int,
        qdrant_client: Any | None = None,
        sqlite_store: SqliteStore | None = None,
    ) -> None:
        self.qdrant = qdrant_client or QdrantClient(url=qdrant_url)
        self.sqlite = sqlite_store or SqliteStore(sqlite_path)
        self.embed_dim = embed_dim

    def add_unit(self, unit: MemoryUnit) -> None:
        vector = embedding.get_embedding(unit.text, prefix=_state.EMBED_PREFIX_DOC)
        self._upsert_units(unit.bank_id, [unit], [vector])

    def batch_add_units(self, units: list[MemoryUnit]) -> None:
        if not units:
            return
        vectors = [embedding.get_embedding(unit.text, prefix=_state.EMBED_PREFIX_DOC) for unit in units]
        units_by_bank: dict[str, list[tuple[MemoryUnit, list[float]]]] = {}
        for unit, vector in zip(units, vectors, strict=True):
            units_by_bank.setdefault(unit.bank_id, []).append((unit, vector))
        for bank_id, grouped in units_by_bank.items():
            bank_units = [unit for unit, _ in grouped]
            bank_vectors = [vector for _, vector in grouped]
            self._upsert_units(bank_id, bank_units, bank_vectors)

    def upsert_units_with_vectors(self, units: list[MemoryUnit], vectors: list[list[float]]) -> None:
        if not units:
            return
        units_by_bank: dict[str, list[tuple[MemoryUnit, list[float]]]] = {}
        for unit, vector in zip(units, vectors, strict=True):
            units_by_bank.setdefault(unit.bank_id, []).append((unit, vector))
        for bank_id, grouped in units_by_bank.items():
            bank_units = [unit for unit, _ in grouped]
            bank_vectors = [vector for _, vector in grouped]
            self._upsert_units(bank_id, bank_units, bank_vectors)

    def add_link(self, link: MemoryLink) -> None:
        self.sqlite.add_link(link)

    def batch_add_links(self, links: list[MemoryLink]) -> None:
        self.sqlite.batch_add_links(links)

    def get_unit(self, unit_id: str, bank_id: str) -> MemoryUnit | None:
        collection_name = self._collection_name(bank_id)
        try:
            points = self.qdrant.retrieve(
                collection_name=collection_name,
                ids=[unit_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return None
        if not points:
            return None
        payload = dict(points[0].payload or {})
        payload["id"] = str(points[0].id)
        payload.setdefault("bank_id", bank_id)
        return model_validate_compat(MemoryUnit, payload)

    def resolve_entity(self, name: str, bank_id: str) -> Entity | None:
        return self.sqlite.resolve_entity(name, bank_id)

    def upsert_entity(self, entity: Entity) -> None:
        self.sqlite.upsert_entity(entity)

    def expand_links(
        self,
        unit_ids: list[str],
        bank_id: str,
        link_types: list[str],
        limit_per_source: int = 10,
    ) -> list[MemoryLink]:
        return self.sqlite.expand_links(unit_ids, bank_id, link_types, limit_per_source=limit_per_source)

    def ensure_collection(self, bank_id: str) -> None:
        collection_name = self._collection_name(bank_id)
        try:
            self.qdrant.get_collection(collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE),
            )
        try:
            self.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="bank_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def init_schema(self) -> None:
        self.sqlite.init_schema()

    def _collection_name(self, bank_id: str) -> str:
        return f"memory_units_{bank_id}"

    def _upsert_units(self, bank_id: str, units: list[MemoryUnit], vectors: list[list[float]]) -> None:
        self.init_schema()
        self.ensure_collection(bank_id)
        collection_name = self._collection_name(bank_id)
        self.qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=unit.id, vector=vector, payload=self._unit_payload(unit))
                for unit, vector in zip(units, vectors, strict=True)
            ],
        )
        self._sync_unit_entities(units)

    def _sync_unit_entities(self, units: list[MemoryUnit]) -> None:
        joins: list[EntityUnitJoin] = []
        for unit in units:
            mention_time = unit.mentioned_at or unit.event_date or unit.occurred_start or datetime.now(timezone.utc)
            for entity_ref in unit.entities:
                entity = Entity(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{unit.bank_id}:{entity_ref.name.lower()}")),
                    bank_id=unit.bank_id,
                    canonical_name=entity_ref.name,
                    aliases=[],
                    entity_type=entity_ref.type,
                    first_mentioned_at=mention_time,
                    last_mentioned_at=mention_time,
                    mention_count=1,
                )
                self.sqlite.upsert_entity(entity)
                stored_entity = self.sqlite.get_entity(entity.canonical_name, entity.bank_id)
                if stored_entity is None:
                    continue
                joins.append(
                    EntityUnitJoin(
                        entity_id=stored_entity.id,
                        unit_id=unit.id,
                        role=entity_ref.role or "other",
                        bank_id=unit.bank_id,
                    )
                )
        self.sqlite.batch_add_entity_units(joins)

    def _unit_payload(self, unit: MemoryUnit) -> dict[str, Any]:
        payload = model_dump_compat(unit)
        payload.pop("id", None)
        return payload
