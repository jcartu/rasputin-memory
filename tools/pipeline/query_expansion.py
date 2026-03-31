import json
import os
from typing import Any

ENTITY_GRAPH_PATH = os.environ.get(
    "ENTITY_GRAPH_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "entity_graph.json"),
)

_KNOWN_ENTITIES_PATH = os.environ.get(
    "KNOWN_ENTITIES_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "known_entities.json"),
)


def lookup_entity_graph(name: str, entity_graph_path: str | None = None) -> str:
    graph_path = entity_graph_path or ENTITY_GRAPH_PATH
    try:
        with open(graph_path) as f:
            graph_data: dict[str, Any] = json.load(f)
    except Exception:
        return ""

    name_lower = name.lower()
    for person, data in graph_data.get("people", {}).items():
        if name_lower in person.lower():
            return f"{data.get('role', '')} {data.get('context', '')}".strip()
    for company, data in graph_data.get("companies", {}).items():
        if name_lower in company.lower():
            return f"{data.get('type', '')} {data.get('context', '')}".strip()
    return ""


def _lookup_known_entities(query: str) -> list[str]:
    try:
        with open(_KNOWN_ENTITIES_PATH) as f:
            data: dict[str, Any] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    query_lower = query.lower()
    hits: list[str] = []
    for name in data.get("persons", []) + data.get("organizations", []) + data.get("projects", []):
        if name.lower() in query_lower:
            hits.append(name)
    return hits


def expand_queries(query: str, max_expansions: int = 5) -> list[str]:
    queries = [query]

    for entity in _lookup_known_entities(query)[:4]:
        graph_context = lookup_entity_graph(entity)
        candidate = f"{entity} {graph_context}".strip() if graph_context else entity
        if candidate and candidate not in queries:
            queries.append(candidate)

    return queries[: max(1, max_expansions)]
