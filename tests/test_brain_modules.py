import importlib
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


config = importlib.import_module("config")
state = importlib.import_module("brain._state")
entities = importlib.import_module("brain.entities")
embedding = importlib.import_module("brain.embedding")
amac = importlib.import_module("brain.amac")
scoring = importlib.import_module("brain.scoring")
graph = importlib.import_module("brain.graph")
commit_module = importlib.import_module("brain.commit")
search_module = importlib.import_module("brain.search")
server = importlib.import_module("brain.server")


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self._text = text
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def test_state_loads_config_correctly():
    cfg = config.load_config("config/rasputin.toml")
    assert state.CONFIG["qdrant"]["collection"] == cfg["qdrant"]["collection"]
    assert state.SERVER_HOST == cfg["server"]["host"]
    assert state.SERVER_PORT == int(cfg["server"]["port"])


def test_extract_entities_fast_various_inputs(monkeypatch):
    monkeypatch.setattr(
        entities,
        "_load_known_entities",
        lambda: ({"John Doe"}, {"OpenClaw"}, {"Rasputin Memory"}),
    )
    text = "John Doe shared Rasputin Memory updates with OpenClaw and Jane Smith"
    extracted = entities.extract_entities_fast(text)

    assert ("John Doe", "Person") in extracted
    assert ("OpenClaw", "Organization") in extracted
    assert ("Rasputin Memory", "Project") in extracted
    assert ("Jane Smith", "Person") in extracted


def test_embedding_check_duplicate_with_mocked_qdrant(monkeypatch):
    near = SimpleNamespace(id=501, score=0.96, payload={"text": "John Doe moved to Toronto yesterday"})
    monkeypatch.setattr(state, "qdrant", SimpleNamespace(query_points=lambda **_k: SimpleNamespace(points=[near])))

    is_dupe, existing_id, similarity = embedding.check_duplicate([0.1] * 768, "John Doe moved to Toronto yesterday")
    assert is_dupe is True
    assert existing_id == 501
    assert similarity >= 0.95


def test_embedding_check_duplicate_rejects_low_overlap(monkeypatch):
    candidate = SimpleNamespace(id=777, score=0.93, payload={"text": "Completely unrelated business update"})
    monkeypatch.setattr(state, "qdrant", SimpleNamespace(query_points=lambda **_k: SimpleNamespace(points=[candidate])))

    is_dupe, existing_id, similarity = embedding.check_duplicate([0.1] * 768, "Health appointment details with doctor")
    assert is_dupe is False
    assert existing_id is None
    assert similarity == 0


def test_amac_score_parser_handles_sentinel_and_fallback(monkeypatch):
    monkeypatch.setattr(state.requests, "post", lambda *a, **k: FakeResponse("SCORES: 8,7,6"))
    assert amac.amac_score("memory") == (8.0, 7.0, 6.0, 7.0)

    raw = "example 0,1,0\nexample 4,2,2\nfinal answer\n9,8,7"
    monkeypatch.setattr(state.requests, "post", lambda *a, **k: FakeResponse(raw))
    assert amac.amac_score("memory") == (9.0, 8.0, 7.0, 8.0)


def test_apply_temporal_decay_math():
    now = datetime.now(timezone.utc)
    rows = [{"score": 1.0, "date": (now - timedelta(days=14)).isoformat(), "importance": 20, "retrieval_count": 0}]
    out = scoring.apply_temporal_decay(rows)

    half_life = 14
    stability = half_life / math.log(2)
    decay_factor = math.exp(-14 / stability)
    expected = round(1.0 * (0.2 + 0.8 * decay_factor), 4)
    assert out[0]["score"] == expected


def test_apply_multifactor_scoring_composite(monkeypatch):
    monkeypatch.setattr(scoring, "get_source_weight", lambda _source: 0.8)
    rows = [{"score": 0.8, "importance": 80, "source": "conversation", "retrieval_count": 5, "days_old": 3}]
    out = scoring.apply_multifactor_scoring(rows)

    multiplier = 0.35 + 0.25 * 0.8 + 0.20 * 1.0 + 0.10 * 0.8 + 0.10 * 0.5
    assert out[0]["multifactor"] == round(multiplier, 3)
    assert out[0]["score"] == round(0.8 * multiplier, 4)


def test_graph_write_to_graph_with_mocked_redis(monkeypatch, mock_redis):
    monkeypatch.setattr(state, "FALKORDB_DISABLED", False)
    monkeypatch.setattr(state, "get_redis", lambda: mock_redis)

    ok, connected = graph.write_to_graph(
        point_id=42,
        text="John Doe updated OpenClaw roadmap",
        entities=[("John Doe", "Person"), ("OpenClaw", "Organization")],
        timestamp=datetime.now().isoformat(),
    )

    assert ok is True
    assert connected == ["John Doe", "OpenClaw"]
    assert len(mock_redis.calls) == 3


def test_commit_memory_full_pipeline_with_mocks(monkeypatch, mock_qdrant):
    monkeypatch.setattr(state, "qdrant", mock_qdrant)
    monkeypatch.setattr(embedding, "get_embedding", lambda *a, **k: [0.2] * 768)
    monkeypatch.setattr(embedding, "check_duplicate", lambda *a, **k: (False, None, 0))
    monkeypatch.setattr(commit_module, "check_contradictions", lambda *a, **k: [])
    monkeypatch.setattr(entities, "extract_entities_fast", lambda *_: [("Jane Smith", "Person")])
    monkeypatch.setattr(graph, "write_to_graph", lambda *a, **k: (True, ["Jane Smith"]))

    result = commit_module.commit_memory(
        "Jane Smith confirmed launch date for project rollout and stakeholder sync is tomorrow morning.",
        source="conversation",
        importance=70,
        metadata={"custom_tag": "release"},
    )

    assert result["ok"] is True
    assert result["dedup"]["action"] == "created"
    pid = result["id"]
    payload = mock_qdrant.points[pid]["payload"]
    assert payload["source"] == "conversation"
    assert payload["importance"] == 70
    assert payload["custom_tag"] == "release"
    assert payload["connected_to"] == ["Jane Smith"]


def test_hybrid_search_with_mocks(monkeypatch):
    monkeypatch.setattr(search_module, "expand_queries", lambda *_a, **_k: ["q1", "q2"])
    monkeypatch.setattr(
        search_module,
        "qdrant_search",
        lambda q, **_k: [
            {
                "score": 0.7 if q == "q1" else 0.9,
                "text": "same memory text",
                "source": "conversation",
                "date": "",
                "importance": 60,
                "retrieval_count": 0,
                "point_id": q,
                "origin": "qdrant",
            }
        ],
    )
    monkeypatch.setattr(graph, "graph_search", lambda *_a, **_k: [])
    monkeypatch.setattr(graph, "enrich_with_graph", lambda *_a, **_k: {})
    monkeypatch.setattr(search_module, "_update_access_tracking", lambda *_a, **_k: None)
    monkeypatch.setattr(embedding, "is_reranker_available", lambda: False)
    monkeypatch.setattr(state, "BM25_AVAILABLE", False)

    out = search_module.hybrid_search("query", limit=5, expand=True)
    assert len(out["results"]) == 1
    assert out["results"][0]["score"] == 0.9
    assert out["stats"]["expanded_queries"] == 2


def test_simple_rate_limiter_allows_and_blocks(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(server.time, "time", lambda: now["t"])

    limiter = server.SimpleRateLimiter(calls_per_minute=2)
    assert limiter.allow("/search:127.0.0.1") is True
    assert limiter.allow("/search:127.0.0.1") is True
    assert limiter.allow("/search:127.0.0.1") is False

    now["t"] += 61
    assert limiter.allow("/search:127.0.0.1") is True
