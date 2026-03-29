#!/usr/bin/env python3
"""FalkorDB schema setup for the graph brain layer."""

from falkordb import FalkorDB

GRAPH_NAME = "brain"
FALKOR_HOST = "localhost"
FALKOR_PORT = 6380


def get_graph():
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    return db.select_graph(GRAPH_NAME)


def create_schema():
    g = get_graph()

    # Indexes for fast lookup
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.id)",
        "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.date)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
        "CREATE INDEX IF NOT EXISTS FOR (o:Organization) ON (o.name)",
        "CREATE INDEX IF NOT EXISTS FOR (pr:Project) ON (pr.name)",
        "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
        "CREATE INDEX IF NOT EXISTS FOR (l:Location) ON (l.name)",
    ]

    for idx in indexes:
        try:
            g.query(idx)
            print(f"  ✓ {idx.split('FOR')[1].strip()}")
        except Exception as e:
            if "already indexed" in str(e).lower() or "exists" in str(e).lower():
                print(f"  - {idx.split('FOR')[1].strip()} (already exists)")
            else:
                print(f"  ✗ {idx}: {e}")

    # Full-text index on Memory text
    try:
        g.query("CALL db.idx.fulltext.createNodeIndex('Memory', 'text')")
        print("  ✓ Full-text index on Memory.text")
    except Exception as e:
        if "already" in str(e).lower() or "exists" in str(e).lower():
            print("  - Full-text index on Memory.text (already exists)")
        else:
            print(f"  Note: Full-text index: {e}")

    print("\nSchema ready.")
    return g


def stats():
    g = get_graph()
    labels = ["Memory", "Person", "Organization", "Project", "Topic", "Location"]
    print("\n📊 Graph Stats:")
    for label in labels:
        try:
            result = g.query(f"MATCH (n:{label}) RETURN count(n) AS c")
            count = result.result_set[0][0]
            print(f"  {label}: {count:,}")
        except:
            print(f"  {label}: 0")

    try:
        result = g.query("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC")
        if result.result_set:
            print("\n  Relationships:")
            for row in result.result_set:
                print(f"    {row[0]}: {row[1]:,}")
    except:
        pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    else:
        print("Creating FalkorDB graph schema...")
        create_schema()
        stats()
