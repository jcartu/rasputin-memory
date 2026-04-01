#!/usr/bin/env python3

import argparse
import os
import sys

import redis


def migrate_mentions_edges(graph_name: str, host: str, port: int, dry_run: bool = False) -> int:
    r = redis.Redis(host=host, port=port)
    r.ping()

    count_query = "MATCH (n)-[r:MENTIONED_IN]->(m:Memory) RETURN count(r)"
    count_result = r.execute_command("GRAPH.QUERY", graph_name, count_query)
    edge_count = int(count_result[1][0][0]) if len(count_result) > 1 and count_result[1] else 0

    if dry_run or edge_count == 0:
        return edge_count

    migrate_query = "MATCH (n)-[r:MENTIONED_IN]->(m:Memory) CREATE (m)-[:MENTIONS]->(n) DELETE r"
    r.execute_command("GRAPH.QUERY", graph_name, migrate_query)
    return edge_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time migration: MENTIONED_IN -> MENTIONS")
    parser.add_argument("--graph", default=os.environ.get("FALKORDB_GRAPH", "brain"))
    parser.add_argument("--host", default=os.environ.get("FALKORDB_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FALKORDB_PORT", "6380")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    converted = migrate_mentions_edges(args.graph, args.host, args.port, dry_run=args.dry_run)
    mode = "would convert" if args.dry_run else "converted"
    print(f"{mode} {converted} MENTIONED_IN edges")
    sys.exit(0)
