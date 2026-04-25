"""Hindsight-compatible graph schemas.

Legacy Rasputin facts use ``fact_type`` values ``world``, ``experience``, and
``inference``. Sprint 1.3 adopts Hindsight's field shape, which expects the
canonical set ``world``, ``experience``, ``observation``, and ``opinion``.
To stay backward-compatible while keeping the new schema stable,
``normalize_fact_type()`` maps legacy ``inference`` to ``observation`` on load.
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, validator

CanonicalFactType = Literal["world", "experience", "observation", "opinion"]
LinkType = Literal[
    "entity_cooccurrence",
    "semantic",
    "causes",
    "caused_by",
    "enables",
    "prevents",
    "temporal",
    "proof",
]

_ENTITY_TYPE_MAP = {
    "person": "PERSON",
    "location": "LOCATION",
    "date": "DATE",
    "event": "EVENT",
    "preference": "PREFERENCE",
    "organization": "ORGANIZATION",
    "thing": "THING",
    "other": "OTHER",
    "project": "THING",
    "topic": "THING",
    "keyword": "THING",
}
_ALLOWED_ENTITY_TYPES = set(_ENTITY_TYPE_MAP.values())


def normalize_fact_type(value: str) -> CanonicalFactType:
    normalized = value.strip().lower()
    if normalized == "inference":
        return "observation"
    if normalized in {"world", "experience", "observation", "opinion"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported fact_type: {value}")


def normalize_entity_type(value: str | None) -> str:
    if not value:
        return "OTHER"
    normalized = _ENTITY_TYPE_MAP.get(value.strip().lower(), value.strip().upper())
    return normalized if normalized in _ALLOWED_ENTITY_TYPES else "OTHER"


def get_configured_embed_dim() -> int:
    env_value = os.environ.get("EMBED_DIM")
    if env_value:
        return int(env_value)

    try:
        config_module = importlib.import_module("config")
        config = config_module.load_config("config/rasputin.toml")
    except Exception:
        return 768

    embeddings = config.get("embeddings", {})
    return int(embeddings.get("dim", embeddings.get("dimensions", 768)))


ModelT = TypeVar("ModelT", bound=BaseModel)


def model_dump_compat(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")(mode="json")
    return json.loads(model.json())


def model_validate_compat(model_cls: type[ModelT], data: Any) -> ModelT:
    if hasattr(model_cls, "model_validate"):
        return getattr(model_cls, "model_validate")(data)
    return model_cls.parse_obj(data)


class EntityRef(BaseModel):
    name: str
    type: str
    role: str | None = None


class MemoryUnit(BaseModel):
    id: str
    bank_id: str
    text: str
    context: str | None = None
    fact_type: CanonicalFactType = "world"
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    where: str | None = None
    entities: list[EntityRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    proof_count: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
    _ingest_commit_sha: str | None = None
    _ingest_config_hash: str | None = None

    @validator("fact_type", pre=True)
    def _normalize_fact_type(cls, value: str) -> CanonicalFactType:
        return normalize_fact_type(value)


class MemoryLink(BaseModel):
    id: str
    bank_id: str
    from_unit_id: str
    to_unit_id: str
    link_type: LinkType
    weight: float = Field(ge=0.0, le=1.0)
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    id: str
    bank_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str = "OTHER"
    first_mentioned_at: datetime | None = None
    last_mentioned_at: datetime | None = None
    mention_count: int = 0

    @validator("entity_type", pre=True)
    def _normalize_entity_type(cls, value: str | None) -> str:
        return normalize_entity_type(value)


class EntityUnitJoin(BaseModel):
    entity_id: str
    unit_id: str
    role: str | None = None
    bank_id: str
