import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

query_expansion = importlib.import_module("pipeline.query_expansion")
source_tiering = importlib.import_module("pipeline.source_tiering")
get_source_weight = source_tiering.get_source_weight
search_module = importlib.import_module("brain.search")
graph_module = importlib.import_module("brain.graph")
embedding_module = importlib.import_module("brain.embedding")
state = importlib.import_module("brain._state")


def test_query_expansion_basic():
    queries = query_expansion.expand_queries(
        "What happened with the project?",
        max_expansions=10,
    )

    assert queries
    assert queries[0] == "What happened with the project?"


def test_query_expansion_entity_aware(tmp_path):
    graph_path = Path(tmp_path) / "entity_graph.json"
    graph_path.write_text(json.dumps({"people": {"Oren": {"role": "founder", "context": "chronos wallet"}}}))
    entities_path = Path(tmp_path) / "known_entities.json"
    entities_path.write_text(json.dumps({"persons": ["Oren"], "organizations": [], "projects": []}))

    setattr(query_expansion, "ENTITY_GRAPH_PATH", str(graph_path))
    setattr(query_expansion, "_KNOWN_ENTITIES_PATH", str(entities_path))
    queries = query_expansion.expand_queries("What did Oren discuss?", max_expansions=10)

    assert any("Oren" in q and "founder" in q for q in queries)


def test_source_tiering_gold():
    assert get_source_weight("conversation") >= 0.9
    assert get_source_weight("email") >= 0.9


def test_source_tiering_unknown_source():
    assert get_source_weight("not_a_real_source") == 0.5


def test_search_uses_query_expansion(monkeypatch):
    calls = []

    def fake_expand_queries(query, max_expansions=5):
        return [query, f"{query} alternate"]

    def fake_qdrant_search(query, limit=10, source_filter=None):
        calls.append(query)
        base_score = 0.8 if "alternate" in query else 0.4
        return [
            {
                "score": base_score,
                "text": "duplicate result",
                "source": "conversation",
                "date": "",
                "title": "",
                "url": "",
                "domain": "",
                "importance": 60,
                "retrieval_count": 0,
                "point_id": query,
                "origin": "qdrant",
            }
        ]

    monkeypatch.setattr(search_module, "expand_queries", fake_expand_queries)
    monkeypatch.setattr(search_module, "qdrant_search", fake_qdrant_search)
    monkeypatch.setattr(graph_module, "graph_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(graph_module, "enrich_with_graph", lambda *args, **kwargs: {})
    monkeypatch.setattr(search_module, "_update_access_tracking", lambda *args, **kwargs: None)
    monkeypatch.setattr(embedding_module, "is_reranker_available", lambda: False)
    monkeypatch.setattr(state, "BM25_AVAILABLE", False)

    result = search_module.hybrid_search("vpn history", limit=5, expand=True)

    assert calls == ["vpn history", "vpn history alternate"]
    assert result["results"][0]["score"] == 0.8
    assert result["stats"]["expanded_queries"] == 2


def test_memory_engine_cli_search(monkeypatch, capsys):
    me = importlib.import_module("memory_engine")

    called = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "score": 0.91,
                        "source": "conversation",
                        "text": "search result text",
                    }
                ]
            }

    def fake_get(url, params=None, timeout=None):
        called["url"] = url
        called["params"] = params
        called["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(me.requests, "get", fake_get)
    exit_code = me.main(["recall", "search", "query"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert called["url"].endswith("/search")
    assert called["params"]["q"] == "search query"
    assert "search result text" in output


def test_memory_engine_cli_commit(monkeypatch, capsys):
    me = importlib.import_module("memory_engine")

    called = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "id": 123}

    def fake_post(url, json=None, timeout=None):
        called["url"] = url
        called["json"] = json
        called["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(me.requests, "post", fake_post)
    exit_code = me.main(["commit", "remember", "this", "message"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert called["url"].endswith("/commit")
    assert called["json"]["text"] == "remember this message"
    assert '"ok": true' in output.lower()
