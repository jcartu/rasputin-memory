from __future__ import annotations

from typing import Any

from brain import _state
from brain import entities


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def _safe_graph_label(label: str) -> str:
    allowed = {"Person", "Organization", "Project", "Topic", "Location"}
    return label if label in allowed else "Entity"


def _cypher_string(value: Any) -> str:
    text = _decode(value)
    text = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _cypher_list(values: list[Any]) -> str:
    return "[" + ", ".join(_cypher_string(value) for value in values) + "]"


def write_to_graph(point_id: int, text: str, entities: list[tuple[str, str]], timestamp: str) -> tuple[bool, list[str]]:
    if _state.FALKORDB_DISABLED:
        return True, []
    try:
        redis_client = _state.get_falkordb()
        redis_client.ping()
    except Exception as error:
        _state.logger.error("Graph commit FalkorDB connection error: %s", error)
        return False, []

    text_preview = text[:500].replace("\n", " ")
    ts = timestamp

    try:
        memory_query = (
            f"MERGE (m:Memory {{id: {_cypher_string(str(point_id))}}}) "
            f"SET m.text = {_cypher_string(text_preview)}, m.created_at = {_cypher_string(ts)}"
        )
        redis_client.execute_command(
            "GRAPH.QUERY",
            _state.GRAPH_NAME,
            memory_query,
        )
    except Exception as error:
        _state.logger.error("Graph commit memory node error: %s", error)
        return False, []

    connected_entities = []
    for name, entity_type in entities:
        try:
            safe_label = (
                entity_type if entity_type in ("Person", "Organization", "Project", "Topic", "Location") else "Entity"
            )
            entity_query = (
                f"MERGE (n:{safe_label} {{name: {_cypher_string(name)}}}) "
                f"ON CREATE SET n.type = {_cypher_string(entity_type)}, n.created_at = {_cypher_string(ts)} "
                f"WITH n MATCH (m:Memory {{id: {_cypher_string(str(point_id))}}}) MERGE (m)-[:MENTIONS]->(n)"
            )
            redis_client.execute_command(
                "GRAPH.QUERY",
                _state.GRAPH_NAME,
                entity_query,
            )
            connected_entities.append(name)
        except Exception as error:
            _state.logger.error("Graph commit entity error for '%s': %s", name, error)

    return True, connected_entities


def graph_search(query: str, hops: int = 2, limit: int = 10) -> list[dict[str, Any]]:
    if _state.FALKORDB_DISABLED:
        return []
    try:
        redis_client = _state.get_falkordb()
        redis_client.ping()
    except Exception as error:
        _state.logger.error("Graph connection error: %s", error)
        return []

    typed_entities = entities.extract_entities_fast(query)
    if not typed_entities:
        typed_entities = [(query, "Unknown")]

    results = []
    seen_memory_ids = set()
    seen_context = set()

    for entity_name, entity_type in typed_entities[:4]:
        if entity_type == "Person":
            labels = ["Person"]
        elif entity_type == "Organization":
            labels = ["Organization"]
        elif entity_type == "Project":
            labels = ["Project"]
        elif entity_type == "Topic":
            labels = ["Topic"]
        elif entity_type == "Keyword":
            labels = []
        else:
            labels = ["Person", "Organization", "Project", "Topic", "Location"]

        for label in labels:
            try:
                safe_label = _safe_graph_label(label)
                entity_name_cypher = _cypher_string(entity_name)
                memory_result = redis_client.execute_command(
                    "GRAPH.QUERY",
                    _state.GRAPH_NAME,
                    "MATCH (m:Memory)-[:MENTIONS]->(n:{label}) "
                    "WHERE toLower(n.name) CONTAINS toLower({entity_name}) "
                    "RETURN m.id, m.text, m.created_at, n.name LIMIT {limit}".format(
                        label=safe_label,
                        entity_name=entity_name_cypher,
                        limit=limit,
                    ),
                )
                for row in memory_result[1] or []:
                    memory_id = _decode(row[0])
                    memory_text = _decode(row[1])
                    memory_date = _decode(row[2])
                    matched_entity = _decode(row[3])
                    if memory_id not in seen_memory_ids and memory_text and len(memory_text) > 10:
                        seen_memory_ids.add(memory_id)
                        results.append(
                            {
                                "text": memory_text,
                                "entity": matched_entity,
                                "date": memory_date,
                                "origin": "graph",
                                "graph_hop": 1,
                                "source": "graph_memory",
                            }
                        )
            except Exception as error:
                _state.logger.error("Graph 1-hop memory error for %s/%s: %s", label, entity_name, error)

        if hops >= 2 and labels:
            for label in labels:
                try:
                    safe_label = _safe_graph_label(label)
                    entity_name_cypher = _cypher_string(entity_name)
                    seen_ids_cypher = _cypher_list(sorted(seen_memory_ids))
                    two_hop = redis_client.execute_command(
                        "GRAPH.QUERY",
                        _state.GRAPH_NAME,
                        "MATCH (m1:Memory)-[:MENTIONS]->(n:{label}) "
                        "MATCH (m1)-[:MENTIONS]->(co) "
                        "WHERE toLower(n.name) CONTAINS toLower({entity_name}) AND n <> co "
                        "WITH DISTINCT co, count(m1) AS shared ORDER BY shared DESC LIMIT 5 "
                        "MATCH (m2:Memory)-[:MENTIONS]->(co) "
                        "WHERE NOT m2.id IN {seen_ids} "
                        "RETURN m2.id, m2.text, m2.created_at, co.name, labels(co)[0] LIMIT {limit}".format(
                            label=safe_label,
                            entity_name=entity_name_cypher,
                            seen_ids=seen_ids_cypher,
                            limit=limit,
                        ),
                    )
                    for row in two_hop[1] or []:
                        memory_id = _decode(row[0])
                        memory_text = _decode(row[1])
                        memory_date = _decode(row[2])
                        connected_name = _decode(row[3])
                        connected_type = _decode(row[4])
                        if memory_id not in seen_memory_ids and memory_text and len(memory_text) > 10:
                            seen_memory_ids.add(memory_id)
                            results.append(
                                {
                                    "text": memory_text,
                                    "entity": entity_name,
                                    "connected_to": connected_name,
                                    "node_type": connected_type,
                                    "date": memory_date,
                                    "origin": "graph",
                                    "graph_hop": 2,
                                    "source": "graph_memory",
                                }
                            )
                except Exception as error:
                    _state.logger.error("Graph 2-hop error for %s/%s: %s", label, entity_name, error)

        if entity_type == "Keyword" and len(entity_name) > 3:
            try:
                entity_name_cypher = _cypher_string(entity_name)
                keyword_result = redis_client.execute_command(
                    "GRAPH.QUERY",
                    _state.GRAPH_NAME,
                    "MATCH (m:Memory) WHERE toLower(m.text) CONTAINS toLower({entity_name}) "
                    "RETURN m.id, m.text, m.created_at LIMIT 5".format(entity_name=entity_name_cypher),
                )
                for row in keyword_result[1] or []:
                    memory_id = _decode(row[0])
                    memory_text = _decode(row[1])
                    memory_date = _decode(row[2])
                    if memory_id not in seen_memory_ids and memory_text and len(memory_text) > 10:
                        seen_memory_ids.add(memory_id)
                        results.append(
                            {
                                "text": memory_text,
                                "entity": entity_name,
                                "date": memory_date,
                                "origin": "graph",
                                "graph_hop": 1,
                                "source": "graph_keyword",
                            }
                        )
            except Exception as error:
                _state.logger.error("Graph keyword search error: %s", error)

        for label in labels:
            try:
                safe_label = _safe_graph_label(label)
                entity_name_cypher = _cypher_string(entity_name)
                context_result = redis_client.execute_command(
                    "GRAPH.QUERY",
                    _state.GRAPH_NAME,
                    "MATCH (n:{label})-[rel]-(connected) "
                    "WHERE toLower(n.name) CONTAINS toLower({entity_name}) "
                    "AND NOT labels(connected)[0] = 'Memory' "
                    "RETURN labels(connected)[0], connected.name, type(rel), n.name LIMIT 8".format(
                        label=safe_label,
                        entity_name=entity_name_cypher,
                    ),
                )
                for row in context_result[1] or []:
                    connected_type = _decode(row[0])
                    connected_name = _decode(row[1])
                    relationship = _decode(row[2])
                    node_name = _decode(row[3])
                    key = f"{node_name}:{connected_name}:{relationship}"
                    if key not in seen_context:
                        seen_context.add(key)
                        results.append(
                            {
                                "text": f"{connected_type}: {connected_name}",
                                "entity": node_name,
                                "connected_to": connected_name,
                                "relationship": relationship,
                                "node_type": connected_type,
                                "origin": "graph",
                                "graph_hop": 1,
                                "source": "graph_context",
                            }
                        )
            except Exception:
                continue

    return results[: limit * 2]


def enrich_with_graph(results: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    try:
        enrichment: dict[str, list[dict[str, Any]]] = {}
        entity_names = set()
        for result in results[: max(1, limit)]:
            text = result.get("text", "")
            for name, _ in entities.extract_entities_fast(text[:500]):
                entity_names.add(name)
        for name in list(entity_names)[:8]:
            related = graph_search(name, hops=2, limit=3)
            if related:
                enrichment[name] = related[:3]
        return enrichment
    except Exception as error:
        _state.logger.error("Graph enrichment non-fatal error: %s", error)
        return {}
