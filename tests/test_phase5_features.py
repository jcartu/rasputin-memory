import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

contradiction = importlib.import_module("tools.pipeline.contradiction")
importance_recalculator = importlib.import_module("tools.importance_recalculator")


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
