import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

contradiction = importlib.import_module("tools.pipeline.contradiction")
importance_recalculator = importlib.import_module("tools.importance_recalculator")
hybrid_brain = importlib.import_module("tools.hybrid_brain")
embedding_health = importlib.import_module("tools.embedding_health")


class _FakePoint:
    def __init__(self, point_id, text, score):
        self.id = point_id
        self.score = score
        self.payload = {"text": text}


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    def __init__(self, points):
        self._points = points

    def query_points(self, **_kwargs):
        return _FakeQueryResult(self._points)


def test_contradiction_detected():
    qdrant = _FakeQdrant([_FakePoint(1, "John Doe moved to Moscow in 2025.", 0.94)])
    rows = contradiction.check_contradictions(
        text="John Doe moved to St. Petersburg in 2026.",
        embedding=[0.1] * 8,
        qdrant_client=qdrant,
        collection="second_brain",
        top_k=5,
    )
    assert len(rows) == 1
    assert rows[0]["existing_id"] == 1


def test_non_contradictory_passes():
    qdrant = _FakeQdrant([_FakePoint(1, "John Doe moved to Moscow in 2025.", 0.96)])
    rows = contradiction.check_contradictions(
        text="John Doe moved to Moscow in 2026 and started a new role.",
        embedding=[0.1] * 8,
        qdrant_client=qdrant,
        collection="second_brain",
        top_k=5,
    )
    assert rows == []


class _FakeScrollPoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class _FakeScrollQdrant:
    def __init__(self, points):
        self.points = points
        self.payload_updates = []
        self.calls = 0

    def scroll(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return self.points, None
        return [], None

    def set_payload(self, **kwargs):
        self.payload_updates.append(kwargs)


def test_importance_recalc_boosts_active():
    now = datetime.now(timezone.utc)
    qdrant = _FakeScrollQdrant(
        [
            _FakeScrollPoint(
                11,
                {
                    "text": "phase-5 contradiction work",
                    "importance": 40,
                    "retrieval_count": 8,
                    "last_accessed": (now - timedelta(days=5)).isoformat(),
                },
            )
        ]
    )
    out = importance_recalculator.recalculate_importance(
        qdrant_client=qdrant,
        collection="second_brain",
        execute=True,
        hot_topics={"phase-5", "contradiction"},
        now=now,
    )
    assert out["updated"] == 1
    assert qdrant.payload_updates[0]["payload"]["importance"] == 55


def test_importance_recalc_decays_stale():
    now = datetime.now(timezone.utc)
    qdrant = _FakeScrollQdrant(
        [
            _FakeScrollPoint(
                22,
                {
                    "text": "old memory",
                    "importance": 50,
                    "retrieval_count": 0,
                    "last_accessed": (now - timedelta(days=120)).isoformat(),
                },
            )
        ]
    )
    out = importance_recalculator.recalculate_importance(
        qdrant_client=qdrant,
        collection="second_brain",
        execute=True,
        hot_topics=set(),
        now=now,
    )
    assert out["updated"] == 1
    assert qdrant.payload_updates[0]["payload"]["importance"] == 40


class _PointWithPayload:
    def __init__(self, payload):
        self.payload = payload


def test_feedback_positive_boosts(monkeypatch):
    state = {"payload": {"importance": 60}}

    class _FeedbackQdrant:
        def retrieve(self, **_kwargs):
            return [_PointWithPayload(state["payload"])]

        def set_payload(self, **kwargs):
            state["update"] = kwargs

    monkeypatch.setattr(hybrid_brain, "qdrant", _FeedbackQdrant())
    result = hybrid_brain.apply_relevance_feedback(point_id="p1", helpful=True)
    assert result["ok"] is True
    assert result["importance_after"] == 65
    updated_payload = state["update"]["payload"]
    assert isinstance(updated_payload, dict)
    assert "last_feedback" in updated_payload


def test_feedback_negative_decays(monkeypatch):
    state = {"payload": {"importance": 60}}

    class _FeedbackQdrant:
        def retrieve(self, **_kwargs):
            return [_PointWithPayload(state["payload"])]

        def set_payload(self, **kwargs):
            state["update"] = kwargs

    monkeypatch.setattr(hybrid_brain, "qdrant", _FeedbackQdrant())
    result = hybrid_brain.apply_relevance_feedback(point_id="p2", helpful=False)
    assert result["ok"] is True
    assert result["importance_after"] == 50
    updated_payload = state["update"]["payload"]
    assert isinstance(updated_payload, dict)
    assert "last_feedback" in updated_payload


def test_graph_enrichment_returns_data(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "extract_entities_fast", lambda _text: [("John Doe", "Person")])
    monkeypatch.setattr(
        hybrid_brain,
        "graph_search",
        lambda *_args, **_kwargs: [
            {
                "text": "John Doe worked with BrandA",
                "origin": "graph",
                "graph_hop": 1,
                "source": "graph_memory",
            }
        ],
    )

    enrichment = hybrid_brain.enrich_with_graph(
        [{"text": "John Doe discussed roadmap"}],
        limit=3,
    )
    assert "John Doe" in enrichment
    assert enrichment["John Doe"][0]["origin"] == "graph"


def test_proactive_surfaces_related_memories(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(hybrid_brain, "extract_entities_fast", lambda _text: [("BrandA", "Organization")])
    monkeypatch.setattr(
        hybrid_brain,
        "qdrant_search",
        lambda *_args, **_kwargs: [
            {
                "text": "BrandA was planning a Europe launch.",
                "score": 0.8,
                "source": "conversation",
                "last_accessed": (now - timedelta(days=10)).isoformat(),
            },
            {
                "text": "BrandA weekly standup notes.",
                "score": 0.9,
                "source": "conversation",
                "last_accessed": (now - timedelta(days=2)).isoformat(),
            },
        ],
    )

    rows = hybrid_brain.proactive_surface(["Can you remind me about BrandA strategy?"], max_results=5)
    assert len(rows) == 1
    assert "Europe launch" in rows[0]["text"]
    assert rows[0]["reason"] == "Related to: BrandA"


def test_embedding_health_no_drift():
    class _Point:
        def __init__(self, point_id, text, vector):
            self.id = point_id
            self.payload = {"text": text}
            self.vector = vector

    class _Qdrant:
        def scroll(self, **_kwargs):
            return [
                _Point(1, "alpha", [1.0, 0.0, 0.0]),
                _Point(2, "beta", [0.0, 1.0, 0.0]),
            ], None

    expected = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
    }
    def embed_fn(text):
        return expected[text]

    out = embedding_health.check_embedding_consistency(
        qdrant_client=_Qdrant(),
        collection="second_brain",
        embed_fn=embed_fn,
        sample_size=2,
        threshold=0.95,
        seed=42,
    )
    assert out["total"] == 2
    assert out["drifted"] == 0
    assert out["drift_rate"] == 0.0
