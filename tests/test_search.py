import importlib
from datetime import datetime, timedelta, timezone


scoring = importlib.import_module("brain.scoring")
search_module = importlib.import_module("brain.search")
graph_module = importlib.import_module("brain.graph")
embedding_module = importlib.import_module("brain.embedding")
state = importlib.import_module("brain._state")


def test_temporal_decay_reduces_old_scores():
    now = datetime.now(timezone.utc)
    rows = [
        {"score": 1.0, "date": (now - timedelta(days=180)).isoformat(), "importance": 20, "retrieval_count": 0},
        {"score": 1.0, "date": (now - timedelta(days=2)).isoformat(), "importance": 20, "retrieval_count": 0},
    ]
    out = scoring.apply_temporal_decay(rows)
    oldest = min(out, key=lambda x: x["days_old"])
    newest = max(out, key=lambda x: x["days_old"])
    assert newest["score"] < oldest["score"]


def test_temporal_decay_preserves_recent():
    now = datetime.now(timezone.utc)
    rows = [{"score": 1.0, "date": (now - timedelta(hours=12)).isoformat(), "importance": 90, "retrieval_count": 5}]
    out = scoring.apply_temporal_decay(rows)
    assert out[0]["score"] > 0.95


def test_multifactor_scoring():
    rows = [
        {"score": 0.8, "importance": 90, "source": "conversation", "retrieval_count": 10, "days_old": 1},
        {"score": 0.8, "importance": 20, "source": "web_page", "retrieval_count": 0, "days_old": 120},
    ]
    out = scoring.apply_multifactor_scoring(rows)
    assert out[0]["importance"] == 90
    assert out[0]["score"] > out[1]["score"]


def test_dedup_removes_same_thread(monkeypatch):
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
    monkeypatch.setattr(graph_module, "graph_search", lambda *_a, **_k: [])
    monkeypatch.setattr(graph_module, "enrich_with_graph", lambda *_a, **_k: {})
    monkeypatch.setattr(search_module, "_update_access_tracking", lambda *_a, **_k: None)
    monkeypatch.setattr(embedding_module, "is_reranker_available", lambda: False)
    monkeypatch.setattr(state, "BM25_AVAILABLE", False)

    out = search_module.hybrid_search("query", limit=5, expand=True)
    assert len(out["results"]) == 1
    assert out["results"][0]["score"] == 0.9


def test_source_tiering_weights():
    rows = [
        {
            "score": 0.7,
            "importance": 60,
            "source": "conversation",
            "source_weight": 0.95,
            "retrieval_count": 0,
            "days_old": 10,
        },
        {
            "score": 0.7,
            "importance": 60,
            "source": "web_page",
            "source_weight": 0.35,
            "retrieval_count": 0,
            "days_old": 10,
        },
    ]
    out = scoring.apply_multifactor_scoring(rows)
    assert out[0]["source"] == "conversation"
    assert out[0]["multifactor"] > out[1]["multifactor"]
