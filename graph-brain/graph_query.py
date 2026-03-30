#!/usr/bin/env python3
"""
Query the FalkorDB graph brain.

Usage:
    python3 graph_query.py "who is connected to ExampleOrg?"
    python3 graph_query.py --entity "Alice Smith" --hops 2
    python3 graph_query.py --topic "igaming" --since "2026-02-01"
    python3 graph_query.py --cypher "MATCH (p:Person) RETURN p.name LIMIT 10"
    python3 graph_query.py --stats
"""

import os
import argparse
import json
import re
from falkordb import FalkorDB

FALKOR_HOST = os.environ.get("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.environ.get("FALKOR_PORT", 6380))
GRAPH_NAME = "brain"

# Destructive Cypher operations that require --force for raw --cypher input
_DESTRUCTIVE_OPS = re.compile(r'\b(DELETE|DROP|REMOVE|DETACH)\b', re.IGNORECASE)


def get_graph():
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    return db.select_graph(GRAPH_NAME)


def entity_neighborhood(graph, entity_name, hops=1, limit=50):
    """Get all nodes within N hops of an entity."""
    hops = min(int(hops), 5)
    results = []
    for label in ["Person", "Organization", "Project", "Topic", "Location"]:
        try:
            q = (
                f"MATCH (start:{label} {{name: $name}}) "
                f"MATCH path = (start)-[*1..{hops}]-(connected) "
                "RETURN DISTINCT labels(connected)[0] AS type, "
                "       connected.name AS name, "
                "       connected.text AS text, "
                "       length(path) AS distance "
                "ORDER BY distance "
                "LIMIT $lim"
            )
            result = graph.query(q, params={"name": entity_name, "lim": limit})
            for row in result.result_set:
                results.append({
                    "type": row[0],
                    "name": row[1] or (row[2][:100] if row[2] else "?"),
                    "distance": row[3]
                })
        except Exception:
            continue

    if not results:
        # Fuzzy search
        for label in ["Person", "Organization", "Project", "Topic", "Location"]:
            try:
                q = (
                    f"MATCH (n:{label}) "
                    "WHERE toLower(n.name) CONTAINS toLower($name) "
                    f"RETURN n.name AS name, '{label}' AS type "
                    "LIMIT 5"
                )
                result = graph.query(q, params={"name": entity_name})
                for row in result.result_set:
                    results.append({"type": row[1], "name": row[0], "distance": 0, "note": "fuzzy match"})
            except Exception:
                continue

    return results


def topic_memories(graph, topic, since=None, limit=20):
    """Get memories about a topic, optionally filtered by date."""
    if since:
        q = (
            "MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: $topic}) "
            "WHERE m.text IS NOT NULL AND m.date >= $since "
            "RETURN m.id, m.text, m.date, m.source "
            "ORDER BY m.date DESC "
            "LIMIT $lim"
        )
        result = graph.query(q, params={"topic": topic.lower(), "since": since, "lim": limit})
    else:
        q = (
            "MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: $topic}) "
            "WHERE m.text IS NOT NULL "
            "RETURN m.id, m.text, m.date, m.source "
            "ORDER BY m.date DESC "
            "LIMIT $lim"
        )
        result = graph.query(q, params={"topic": topic.lower(), "lim": limit})
    return [{"id": r[0], "text": r[1], "date": r[2], "source": r[3]} for r in result.result_set]


def natural_query(graph, question, limit=20):
    """Attempt a natural language query by extracting key terms and searching."""
    words = question.split()
    entities = [w.strip("?.,!\"'") for w in words if w[0:1].isupper() and len(w) > 2]

    results = []
    for entity in entities[:3]:
        neighborhood = entity_neighborhood(graph, entity, hops=2, limit=limit)
        if neighborhood:
            results.append({"entity": entity, "connections": neighborhood})

    topics = [w.lower().strip("?.,!\"'") for w in words if len(w) > 3 and not w[0:1].isupper()]
    for topic in topics[:2]:
        memories = topic_memories(graph, topic, limit=5)
        if memories:
            results.append({"topic": topic, "memories": memories})

    return results


def shortest_path(graph, from_entity, to_entity):
    """Find shortest path between two entities."""
    for label1 in ["Person", "Organization", "Project", "Topic", "Location"]:
        for label2 in ["Person", "Organization", "Project", "Topic", "Location"]:
            try:
                q = (
                    f"MATCH path = shortestPath("
                    f"    (a:{label1} {{name: $from_name}})-[*..5]-(b:{label2} {{name: $to_name}})"
                    f") "
                    "RETURN [n IN nodes(path) | coalesce(n.name, left(n.text, 80))] AS path, "
                    "       [r IN relationships(path) | type(r)] AS rels, "
                    "       length(path) AS hops"
                )
                result = graph.query(q, params={"from_name": from_entity, "to_name": to_entity})
                if result.result_set:
                    row = result.result_set[0]
                    return {"path": row[0], "relationships": row[1], "hops": row[2]}
            except Exception:
                continue
    return None


def graph_stats(graph):
    """Print graph statistics."""
    labels = ["Memory", "Person", "Organization", "Project", "Topic", "Location"]
    print("\n📊 Graph Brain Stats")
    print("=" * 40)
    total_nodes = 0
    for label in labels:
        try:
            result = graph.query(f"MATCH (n:{label}) RETURN count(n)")
            count = result.result_set[0][0]
            total_nodes += count
            print(f"  {label:15s} {count:>8,}")
        except Exception:
            print(f"  {label:15s}        0")

    print(f"  {'TOTAL':15s} {total_nodes:>8,}")
    print()

    try:
        result = graph.query("MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC")
        if result.result_set:
            print("  Relationships:")
            total_rels = 0
            for row in result.result_set:
                total_rels += row[1]
                print(f"    {row[0]:15s} {row[1]:>8,}")
            print(f"    {'TOTAL':15s} {total_rels:>8,}")
    except Exception:
        pass

    # Top entities
    for label, desc in [("Person", "People"), ("Organization", "Orgs"), ("Topic", "Topics")]:
        try:
            result = graph.query(f"""
                MATCH (n:{label})<-[r]-()
                RETURN n.name, count(r) AS mentions
                ORDER BY mentions DESC LIMIT 10
            """)
            if result.result_set:
                print(f"\n  Top {desc}:")
                for row in result.result_set:
                    print(f"    {row[0]:30s} ({row[1]} mentions)")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Query the graph brain")
    parser.add_argument("question", nargs="?", help="Natural language question")
    parser.add_argument("--entity", "-e", help="Entity name to explore")
    parser.add_argument("--hops", type=int, default=2, help="Hops for neighborhood (default: 2)")
    parser.add_argument("--topic", "-t", help="Topic to search")
    parser.add_argument("--since", help="Date filter YYYY-MM-DD")
    parser.add_argument("--cypher", "-c", help="Raw Cypher query (use --force for destructive ops)")
    parser.add_argument("--force", action="store_true", help="Allow destructive Cypher operations (DELETE, DROP, REMOVE)")
    parser.add_argument("--path", nargs=2, metavar=("FROM", "TO"), help="Shortest path between entities")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    graph = get_graph()

    if args.stats:
        graph_stats(graph)
        return

    if args.cypher:
        # WARNING: --cypher accepts raw Cypher input. Validate for destructive operations.
        if _DESTRUCTIVE_OPS.search(args.cypher) and not args.force:
            print("⚠️  Destructive Cypher detected (DELETE/DROP/REMOVE). Use --force to execute.")
            return
        result = graph.query(args.cypher)
        if args.json:
            print(json.dumps(result.result_set, indent=2, default=str))
        else:
            for row in result.result_set:
                print(" | ".join(str(x) for x in row))
        return

    if args.entity:
        results = entity_neighborhood(graph, args.entity, hops=args.hops, limit=args.limit)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n🔗 Neighborhood of '{args.entity}' ({args.hops} hops):")
            for r in results:
                dist = f" (dist={r['distance']})" if 'distance' in r else ""
                note = f" [{r['note']}]" if 'note' in r else ""
                print(f"  [{r['type']}] {r['name']}{dist}{note}")
        return

    if args.topic:
        results = topic_memories(graph, args.topic, since=args.since, limit=args.limit)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n📌 Memories about '{args.topic}':")
            for r in results:
                print(f"  [{r.get('date', '?')}] {r.get('text', '')[:120]}")
        return

    if args.path:
        result = shortest_path(graph, args.path[0], args.path[1])
        if result:
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n🛤️ Path ({result['hops']} hops):")
                for i, (node, rel) in enumerate(zip(result['path'], result['relationships'] + [''])):
                    print(f"  {node}")
                    if rel:
                        print(f"    --[{rel}]-->")
        else:
            print("No path found.")
        return

    if args.question:
        results = natural_query(graph, args.question, limit=args.limit)
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for group in results:
                if "entity" in group:
                    print(f"\n🔗 '{group['entity']}' connections:")
                    for c in group["connections"]:
                        print(f"  [{c['type']}] {c['name']}")
                elif "topic" in group:
                    print(f"\n📌 '{group['topic']}' memories:")
                    for m in group["memories"]:
                        print(f"  [{m.get('date', '?')}] {m.get('text', '')[:120]}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
