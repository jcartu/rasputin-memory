import importlib
from datetime import datetime, timedelta, timezone


hybrid_brain = importlib.import_module("hybrid_brain")


def test_temporal_decay_reduces_old_scores():
    now = datetime.now(timezone.utc)
    rows = [
        {"score": 1.0, "date": (now - timedelta(days=180)).isoformat(), "importance": 20, "retrieval_count": 0},
        {"score": 1.0, "date": (now - timedelta(days=2)).isoformat(), "importance": 20, "retrieval_count": 0},
    ]
    out = hybrid_brain.apply_temporal_decay(rows)
    oldest = min(out, key=lambda x: x["days_old"])
    newest = max(out, key=lambda x: x["days_old"])
    assert newest["score"] < oldest["score"]


def test_temporal_decay_preserves_recent():
    now = datetime.now(timezone.utc)
    rows = [{"score": 1.0, "date": (now - timedelta(hours=12)).isoformat(), "importance": 90, "retrieval_count": 5}]
    out = hybrid_brain.apply_temporal_decay(rows)
    assert out[0]["score"] > 0.95


def test_multifactor_scoring():
    rows = [
        {"score": 0.8, "importance": 90, "source": "conversation", "retrieval_count": 10, "days_old": 1},
        {"score": 0.8, "importance": 20, "source": "web_page", "retrieval_count": 0, "days_old": 120},
    ]
    out = hybrid_brain.apply_multifactor_scoring(rows)
    assert out[0]["importance"] == 90
    assert out[0]["score"] > out[1]["score"]


def test_dedup_removes_same_thread(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "expand_queries", lambda *_a, **_k: ["q1", "q2"])
    monkeypatch.setattr(
        hybrid_brain,
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
    monkeypatch.setattr(hybrid_brain, "graph_search", lambda *_a, **_k: [])
    monkeypatch.setattr(hybrid_brain, "enrich_with_graph", lambda *_a, **_k: {})
    monkeypatch.setattr(hybrid_brain, "_update_access_tracking", lambda *_a, **_k: None)
    monkeypatch.setattr(hybrid_brain, "is_reranker_available", lambda: False)
    monkeypatch.setattr(hybrid_brain, "BM25_AVAILABLE", False)

    out = hybrid_brain.hybrid_search("query", limit=5, expand=True)
    assert len(out["results"]) == 1
    assert out["results"][0]["score"] == 0.9


def test_source_tiering_weights():
    rows = [
        {"score": 0.7, "importance": 60, "source": "conversation", "retrieval_count": 0, "days_old": 10},
        {"score": 0.7, "importance": 60, "source": "web_page", "retrieval_count": 0, "days_old": 10},
    ]
    out = hybrid_brain.apply_multifactor_scoring(rows)
    assert out[0]["source"] == "conversation"
    assert out[0]["multifactor"] > out[1]["multifactor"]
