import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import hybrid_brain


class FakeRedis:
    def __init__(self):
        self.queries = []

    def ping(self):
        return True

    def execute_command(self, *args):
        cypher = args[2]
        self.queries.append(cypher)
        if "RETURN m.id, m.text, m.created_at, n.name" in cypher:
            return [[], [["1", "memory text long enough for graph result", "2026-03-30", "John Doe"]]]
        return [[], []]


def test_ner_consistency(monkeypatch):
    monkeypatch.setattr(
        hybrid_brain,
        "_load_known_entities",
        lambda: ({"John Doe"}, {"BrandA", "OpenClaw"}, {"Rasputin Memory"}),
    )

    text = "John Doe built Rasputin Memory with OpenClaw at BrandA."

    write_entities = hybrid_brain.extract_entities_fast(text)
    read_entities = hybrid_brain.extract_entities_fast(text)

    assert set(write_entities) == set(read_entities)
    assert ("John Doe", "Person") in write_entities
    assert ("OpenClaw", "Organization") in write_entities


def test_graph_write_and_read_consistent(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
    monkeypatch.setattr(hybrid_brain.redis, "Redis", lambda *args, **kwargs: fake_redis)

    ok, _ = hybrid_brain.write_to_graph(
        point_id=123,
        text="John Doe met BrandA at OpenClaw HQ",
        entities=[("John Doe", "Person")],
        timestamp="2026-03-30",
    )
    assert ok is True
    assert any("MERGE (m)-[:MENTIONS]->(n)" in q for q in fake_redis.queries)

    monkeypatch.setattr(hybrid_brain, "extract_entities_fast", lambda _q: [("John Doe", "Person")])
    results = hybrid_brain.graph_search("John Doe", hops=1, limit=3)

    assert any("MATCH (m:Memory)-[:MENTIONS]->(n:Person)" in q for q in fake_redis.queries)
    assert len(results) == 1
    assert results[0]["origin"] == "graph"


def test_graph_results_scored_by_hop(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "BM25_AVAILABLE", False)
    monkeypatch.setattr(hybrid_brain, "qdrant_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        hybrid_brain,
        "graph_search",
        lambda *_args, **_kwargs: [
            {"text": "1-hop graph hit", "origin": "graph", "graph_hop": 1, "source": "graph_memory"},
            {"text": "2-hop graph hit", "origin": "graph", "graph_hop": 2, "source": "graph_memory"},
        ],
    )
    monkeypatch.setattr(hybrid_brain, "is_reranker_available", lambda: False)
    monkeypatch.setattr(hybrid_brain, "enrich_with_graph", lambda *_args, **_kwargs: {})

    result = hybrid_brain.hybrid_search("John Doe", limit=2, expand=False)

    scores_by_text = {item["text"]: item["score"] for item in result["results"]}
    assert scores_by_text["1-hop graph hit"] == 0.8
    assert scores_by_text["2-hop graph hit"] == 0.5
