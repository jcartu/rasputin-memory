from __future__ import annotations

import math
import importlib
import os
import re as _re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from brain import _state
from brain import embedding
from brain import entities
from brain import graph


class MemoryPayload(TypedDict, total=False):
    text: str
    source: str
    source_weight: float
    date: str
    importance: int
    auto_committed: bool
    retrieval_count: int
    last_accessed: str
    embedding_model: str
    schema_version: str
    contradicts: list[Any]
    supersedes: list[Any]
    has_contradictions: bool
    speaker: str
    mentioned_names: list[str]
    has_date: bool
    extracted_dates: list[str]
    connected_to: list[Any]
    constraints: list[dict[str, Any]]
    constraint_summary: str
    pending_archive: bool
    soft_deleted: bool
    pending_delete: bool
    fact_type: str
    occurred_start: str | None
    occurred_end: str | None
    confidence: float


safe_import = importlib.import_module("pipeline._imports").safe_import

_scoring_constants = safe_import("pipeline.scoring_constants", "tools.pipeline.scoring_constants")
get_source_weight = _scoring_constants.get_source_weight

_contradiction = safe_import("pipeline.contradiction", "tools.pipeline.contradiction")
check_contradictions = _contradiction.check_contradictions

_CAPITALIZED_NAME_RE = _scoring_constants.CAPITALIZED_NAME_RE
_NAME_STOPWORDS = _scoring_constants.NAME_STOPWORDS
_DATE_RE = _re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|"
    r"\b(?:yesterday|last\s+\w+|ago|before|after|since)\b",
    _re.IGNORECASE,
)


_STRUCTURED_DATE_RE = _re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,.]?\s+(\d{4})\b|"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,.]?\s+(\d{1,2})[,.]?\s+(\d{4})\b|"
    r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
    _re.IGNORECASE,
)


def _extract_dates(text: str) -> list[str]:
    dates: list[str] = []
    for match in _STRUCTURED_DATE_RE.finditer(text):
        groups = match.groups()
        if groups[0] and groups[1] and groups[2]:
            dates.append(f"{groups[0]} {groups[1]} {groups[2]}")
        elif groups[3] and groups[4] and groups[5]:
            dates.append(f"{groups[4]} {groups[3]} {groups[5]}")
        elif groups[6] and groups[7] and groups[8]:
            dates.append(f"{groups[6]}-{groups[7]}-{groups[8]}")
    return dates[:5]


def _extract_mentioned_names(text: str) -> list[str]:
    matches = _CAPITALIZED_NAME_RE.findall(text)
    return list(dict.fromkeys(m for m in matches if not any(w in _NAME_STOPWORDS for w in m.split())))[:20]


def commit_memory(
    text: str,
    source: str = "conversation",
    importance: int = 60,
    metadata: Optional[dict[str, Any]] = None,
    force: bool = False,
    collection: Optional[str] = None,
) -> dict[str, Any]:
    with _state._commit_lock:
        try:
            vector = embedding.get_embedding(text[:4000], prefix=_state.EMBED_PREFIX_DOC)
        except Exception as error:
            _state.logger.error("commit_memory embedding failed: %s", error)
            return {"ok": False, "error": f"Embedding failed: {error}"}

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude < 0.1:
            _state.logger.warning("commit_memory rejected embedding magnitude %.4f too low", magnitude)
            return {"ok": False, "error": f"Embedding magnitude too low: {magnitude:.4f}"}

        is_dupe, existing_id, similarity = embedding.check_duplicate(vector, text)
        dedup_action = "created"

        if is_dupe and existing_id is not None:
            point_id = existing_id
            dedup_action = "updated"
            _state.logger.info("Dedup near-duplicate found (sim=%s), updating point %s", similarity, point_id)
        else:
            point_id = uuid.uuid4().int >> 65

        contradiction_hits: list[dict[str, Any]] = []
        contradiction_ids: list[Any] = []
        supersedes_ids: list[Any] = []
        if not is_dupe:
            contradiction_hits = check_contradictions(
                text=text,
                embedding=vector,
                qdrant_client=_state.qdrant,
                collection=_state.COLLECTION,
                top_k=5,
            )
            contradiction_ids = [
                hit.get("existing_id") for hit in contradiction_hits if hit.get("existing_id") is not None
            ]
            if contradiction_ids:
                supersedes_ids = list(contradiction_ids)

        timestamp = datetime.now(timezone.utc).isoformat()

        payload: MemoryPayload = {
            "text": text[:4000],
            "source": source,
            "source_weight": get_source_weight(source),
            "date": timestamp,
            "importance": importance,
            "auto_committed": True,
            "retrieval_count": 0,
            "embedding_model": _state.EMBED_MODEL,
            "schema_version": "0.8",
            "contradicts": contradiction_ids,
            "supersedes": supersedes_ids,
            "has_contradictions": bool(contradiction_ids),
            "speaker": (metadata or {}).get("speaker", ""),
            "mentioned_names": _extract_mentioned_names(text),
            "has_date": bool(_DATE_RE.search(text)),
            "extracted_dates": _extract_dates(text),
        }
        graph_entities_resolved: list[tuple[str, str]] | None = None
        if os.environ.get("ENTITY_RESOLVER", "0") == "1":
            from brain import entity_resolver

            raw_entities = entities.extract_entities_fast(text)
            resolved = entity_resolver.resolve(raw_entities, text)
            canonical_names = list(dict.fromkeys(canon for _, canon, _ in resolved))
            payload["mentioned_names"] = canonical_names
            graph_entities_resolved = [(canon, etype) for _, canon, etype in resolved]

        if metadata and isinstance(metadata, dict):
            protected_fields = {
                "text",
                "source",
                "source_weight",
                "date",
                "importance",
                "auto_committed",
                "retrieval_count",
                "embedding_model",
                "schema_version",
                "has_contradictions",
                "mentioned_names",
                "has_date",
                "extracted_dates",
                "speaker",
                "connected_to",
                "contradicts",
                "supersedes",
                "pending_archive",
                "soft_deleted",
                "pending_delete",
                "last_accessed",
                "constraints",
                "constraint_summary",
                "fact_type",
                "occurred_start",
                "occurred_end",
                "confidence",
            }
            safe_metadata = {key: value for key, value in metadata.items() if key not in protected_fields}
            payload.update(safe_metadata)  # type: ignore[typeddict-item]

            if "fact_type" in metadata:
                payload["fact_type"] = metadata.get("fact_type", "world")
                payload["occurred_start"] = metadata.get("occurred_start")
                payload["occurred_end"] = metadata.get("occurred_end")
                payload["confidence"] = metadata.get("confidence", 0.8)

        constraint_texts: list[str] = []
        try:
            from brain import constraints as _constraints_mod

            if _constraints_mod.CONSTRAINTS_ENABLED:
                extracted = _constraints_mod.extract_constraints(text)
                if extracted:
                    constraint_texts = [c.get("constraint", "") for c in extracted]
                    payload["constraints"] = extracted
                    payload["constraint_summary"] = " | ".join(constraint_texts)

                    constraint_combined = " | ".join(constraint_texts)
                    try:
                        constraint_vec = embedding.get_embedding(constraint_combined, prefix=_state.EMBED_PREFIX_DOC)
                        _state.qdrant.upsert(
                            collection_name=_state.CONSTRAINT_COLLECTION,
                            points=[
                                PointStruct(
                                    id=point_id,
                                    vector=constraint_vec,
                                    payload={
                                        "constraint_summary": constraint_combined,
                                        "constraints": extracted,
                                        "parent_text": text[:500],
                                        "source": source,
                                        "date": timestamp,
                                    },
                                )
                            ],
                        )
                    except Exception as ce:
                        _state.logger.debug("Constraint collection upsert failed: %s", ce)
        except Exception as exc:
            _state.logger.debug("Constraint extraction skipped: %s", exc)

        try:
            _state.qdrant.upsert(
                collection_name=collection or _state.COLLECTION,
                points=[PointStruct(id=point_id, vector=vector, payload=dict(payload))],
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}

        graph_ok = False
        graph_entities = 0
        connected_to: list[str] = []
        graph_error: str | None = None
        try:
            if graph_entities_resolved is not None:
                extracted_entities = graph_entities_resolved
            else:
                extracted_entities = entities.extract_entities_fast(text)
            graph_entities = len(extracted_entities)
            if extracted_entities:
                graph_result = graph.write_to_graph(point_id, text, extracted_entities, timestamp)
                if isinstance(graph_result, tuple):
                    graph_ok, connected_to = graph_result
                else:
                    graph_ok = graph_result
                if not graph_ok:
                    graph_error = "graph_write_returned_false"
            else:
                graph_ok = True
        except Exception as error:
            graph_error = str(error)
            _state.logger.warning("Graph commit non-fatal error: %s", error)

        if connected_to:
            try:
                _state.qdrant.set_payload(
                    collection_name=_state.COLLECTION,
                    points=[point_id],
                    payload={"connected_to": connected_to},
                )
            except Exception as error:
                _state.logger.error("Graph commit failed to update connected_to payload: %s", error)

        warnings: list[str] = []
        if graph_error:
            warnings.append(f"graph_write_failed: {graph_error}")

        response = {
            "ok": True,
            "id": point_id,
            "source": source,
            "dedup": {"action": dedup_action, "similarity": similarity if is_dupe else None},
            "contradictions": contradiction_hits,
            "graph": {"written": graph_ok, "entities": graph_entities, "connected_to": connected_to},
        }
        if warnings:
            response["warnings"] = warnings
        return response


def list_contradictions(limit: int = 100) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    offset = None
    try:
        while len(entries) < limit:
            points, offset = _state.qdrant.scroll(
                collection_name=_state.COLLECTION,
                scroll_filter=Filter(must=[FieldCondition(key="has_contradictions", match=MatchValue(value=True))]),
                limit=min(100, max(limit - len(entries), 1)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points or []:
                payload = point.payload or {}
                contradicts = payload.get("contradicts") or []
                if contradicts:
                    entries.append(
                        {
                            "point_id": point.id,
                            "text": payload.get("text", "")[:200],
                            "source": payload.get("source", ""),
                            "date": payload.get("date", ""),
                            "contradicts": contradicts,
                            "supersedes": payload.get("supersedes", []),
                        }
                    )
                    if len(entries) >= limit:
                        break
            if offset is None:
                break
    except Exception as error:
        _state.logger.error("Failed to list contradictions: %s", error)
        return []
    return entries


def apply_relevance_feedback(point_id: Any, helpful: bool) -> dict[str, Any]:
    points = _state.qdrant.retrieve(collection_name=_state.COLLECTION, ids=[point_id], with_payload=True)
    if not points:
        return {"ok": False, "error": "point_not_found", "point_id": point_id}

    payload = points[0].payload or {}
    current_importance = payload.get("importance", 50)
    try:
        current_importance = int(current_importance)
    except (TypeError, ValueError):
        current_importance = 50

    if helpful:
        new_importance = min(100, current_importance + 5)
    else:
        new_importance = max(0, current_importance - 10)

    _state.qdrant.set_payload(
        collection_name=_state.COLLECTION,
        points=[point_id],
        payload={
            "importance": new_importance,
            "last_feedback": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "ok": True,
        "point_id": point_id,
        "helpful": helpful,
        "importance_before": current_importance,
        "importance_after": new_importance,
    }


def commit_conversation_turns(
    turns: list[dict],
    source: str = "conversation",
    metadata: dict | None = None,
    collection: str | None = None,
    window_size: int = 5,
    stride: int = 2,
) -> dict:
    results = []
    for turn in turns:
        text = turn.get("text", "")
        if not text:
            continue
        turn_meta = dict(metadata or {})
        turn_meta["speaker"] = turn.get("speaker", "")
        turn_meta["chunk_type"] = "turn"
        result = commit_memory(text, source=source, metadata=turn_meta, collection=collection)
        results.append(result)

    windows_committed = 0
    if len(turns) >= window_size:
        for i in range(0, max(len(turns) - window_size + 1, 1), stride):
            window = turns[i : i + window_size]
            window_texts = [t.get("text", "") for t in window if t.get("text")]
            if not window_texts:
                continue
            combined = "\n".join(window_texts)
            win_meta = dict(metadata or {})
            win_meta["chunk_type"] = "window"
            win_meta["speakers"] = list({t.get("speaker", "") for t in window if t.get("speaker")})
            result = commit_memory(combined, source=source, metadata=win_meta, collection=collection)
            if result.get("ok"):
                windows_committed += 1

    return {
        "ok": True,
        "turns_committed": sum(1 for r in results if r.get("ok")),
        "windows_committed": windows_committed,
        "total": len(results) + windows_committed,
    }
