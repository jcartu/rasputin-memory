from __future__ import annotations

import math
import importlib
import re as _re
import uuid
from datetime import datetime
from typing import Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from brain import _state
from brain import embedding
from brain import entities
from brain import graph

safe_import = importlib.import_module("pipeline._imports").safe_import

_scoring_constants = safe_import("pipeline.scoring_constants", "tools.pipeline.scoring_constants")
get_source_weight = _scoring_constants.get_source_weight

_contradiction = safe_import("pipeline.contradiction", "tools.pipeline.contradiction")
check_contradictions = _contradiction.check_contradictions

_CAPITALIZED_NAME_RE = _re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b")
_NAME_STOPWORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "What",
        "When",
        "Where",
        "Who",
        "How",
        "Yes",
        "Not",
        "But",
        "And",
        "Also",
        "Just",
        "Very",
        "Really",
        "Session",
        "Unknown",
        "None",
        "True",
        "False",
        "Error",
        "Warning",
        "Memory",
        "Search",
        "Query",
        "Answer",
    }
)
_DATE_RE = _re.compile(
    r"\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|"
    r"\b(?:yesterday|last\s+\w+|ago|before|after|since)\b",
    _re.IGNORECASE,
)


def _extract_mentioned_names(text: str) -> list[str]:
    matches = _CAPITALIZED_NAME_RE.findall(text)
    return list(dict.fromkeys(m for m in matches if m not in _NAME_STOPWORDS))[:20]


def commit_memory(
    text: str,
    source: str = "conversation",
    importance: int = 60,
    metadata: Optional[dict[str, Any]] = None,
    force: bool = False,
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

        timestamp = datetime.now().isoformat()

        payload = {
            "text": text[:4000],
            "source": source,
            "source_weight": get_source_weight(source),
            "date": timestamp,
            "importance": importance,
            "auto_committed": True,
            "retrieval_count": 0,
            "embedding_model": _state.EMBED_MODEL,
            "schema_version": "0.3",
            "contradicts": contradiction_ids,
            "supersedes": supersedes_ids,
            "has_contradictions": bool(contradiction_ids),
            "speaker": (metadata or {}).get("speaker", ""),
            "mentioned_names": _extract_mentioned_names(text),
            "has_date": bool(_DATE_RE.search(text)),
        }
        if metadata and isinstance(metadata, dict):
            protected_fields = {
                "text",
                "source",
                "date",
                "importance",
                "auto_committed",
                "retrieval_count",
                "embedding_model",
                "schema_version",
            }
            safe_metadata = {key: value for key, value in metadata.items() if key not in protected_fields}
            payload.update(safe_metadata)

        try:
            _state.qdrant.upsert(
                collection_name=_state.COLLECTION,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}

        graph_ok = False
        graph_entities = 0
        connected_to: list[str] = []
        graph_error: str | None = None
        try:
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
            "last_feedback": datetime.now().isoformat(),
        },
    )

    return {
        "ok": True,
        "point_id": point_id,
        "helpful": helpful,
        "importance_before": current_importance,
        "importance_after": new_importance,
    }
