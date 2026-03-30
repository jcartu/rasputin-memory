import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

contradiction = importlib.import_module("tools.pipeline.contradiction")


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
