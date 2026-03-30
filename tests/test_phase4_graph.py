import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

state = importlib.import_module("brain._state")
entities = importlib.import_module("brain.entities")
graph_module = importlib.import_module("brain.graph")
search_module = importlib.import_module("brain.search")
embedding_module = importlib.import_module("brain.embedding")


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
        entities,
        "_load_known_entities",
        lambda: ({"John Doe"}, {"BrandA", "OpenClaw"}, {"Rasputin Memory"}),
    )

    text = "John Doe built Rasputin Memory with OpenClaw at BrandA."

    write_entities = entities.extract_entities_fast(text)
    read_entities = entities.extract_entities_fast(text)

    assert set(write_entities) == set(read_entities)
    assert ("John Doe", "Person") in write_entities
    assert ("OpenClaw", "Organization") in write_entities


def test_graph_write_and_read_consistent(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(state, "FALKORDB_DISABLED", False)
    monkeypatch.setattr(state.redis, "Redis", lambda *args, **kwargs: fake_redis)

    ok, _ = graph_module.write_to_graph(
        123, "John Doe met BrandA at OpenClaw HQ", [("John Doe", "Person")], "2026-03-30"
    )
    assert ok is True
    assert any("MERGE (m)-[:MENTIONS]->(n)" in q for q in fake_redis.queries)

    monkeypatch.setattr(entities, "extract_entities_fast", lambda _q: [("John Doe", "Person")])
    results = graph_module.graph_search("John Doe", hops=1, limit=3)

    assert any("MATCH (m:Memory)-[:MENTIONS]->(n:Person)" in q for q in fake_redis.queries)
    assert len(results) == 1
    assert results[0]["origin"] == "graph"


def test_graph_results_scored_by_hop(monkeypatch):
    monkeypatch.setattr(state, "BM25_AVAILABLE", False)
    monkeypatch.setattr(search_module, "qdrant_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        graph_module,
        "graph_search",
        lambda *_args, **_kwargs: [
            {"text": "1-hop graph hit", "origin": "graph", "graph_hop": 1, "source": "graph_memory"},
            {"text": "2-hop graph hit", "origin": "graph", "graph_hop": 2, "source": "graph_memory"},
        ],
    )
    monkeypatch.setattr(embedding_module, "is_reranker_available", lambda: False)
    monkeypatch.setattr(graph_module, "enrich_with_graph", lambda *_args, **_kwargs: {})

    result = search_module.hybrid_search("John Doe", limit=2, expand=False)

    scores_by_text = {item["text"]: item["score"] for item in result["results"]}
    assert scores_by_text["1-hop graph hit"] == 0.8
    assert scores_by_text["2-hop graph hit"] == 0.5
