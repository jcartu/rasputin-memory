import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

query_expansion = importlib.import_module("pipeline.query_expansion")
source_tiering = importlib.import_module("pipeline.source_tiering")
get_source_weight = source_tiering.get_source_weight


def test_query_expansion_basic():
    queries = query_expansion.expand_queries(
        "What did we email about crypto last week?",
        max_expansions=10,
    )

    assert queries
    assert queries[0] == "What did we email about crypto last week?"
    assert any(q.startswith("email ") for q in queries)
    assert any(q.startswith("recent ") for q in queries)
    assert any("Bitcoin CHRONOS wallet blockchain hardware" in q for q in queries)


def test_query_expansion_entity_aware(tmp_path):
    graph_path = Path(tmp_path) / "entity_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "people": {
                    "Oren": {
                        "role": "founder",
                        "context": "chronos wallet",
                    }
                }
            }
        )
    )

    setattr(query_expansion, "ENTITY_GRAPH_PATH", str(graph_path))
    queries = query_expansion.expand_queries("What did Oren discuss?", max_expansions=10)

    assert any("Oren founder chronos wallet" in q for q in queries)


def test_source_tiering_gold():
    assert get_source_weight("conversation") >= 0.9
    assert get_source_weight("email") >= 0.9


def test_source_tiering_unknown_source():
    assert get_source_weight("not_a_real_source") == 0.5


def test_search_uses_query_expansion(monkeypatch):
    hb = importlib.import_module("hybrid_brain")

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

    monkeypatch.setattr(hb, "expand_queries", fake_expand_queries)
    monkeypatch.setattr(hb, "qdrant_search", fake_qdrant_search)
    monkeypatch.setattr(hb, "graph_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(hb, "enrich_with_graph", lambda *args, **kwargs: {})
    monkeypatch.setattr(hb, "_update_access_tracking", lambda *args, **kwargs: None)
    monkeypatch.setattr(hb, "is_reranker_available", lambda: False)
    monkeypatch.setattr(hb, "BM25_AVAILABLE", False)

    result = hb.hybrid_search("vpn history", limit=5, expand=True)

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
