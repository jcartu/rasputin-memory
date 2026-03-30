import importlib
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


memory_decay = importlib.import_module("memory_decay")
hybrid_brain = importlib.import_module("hybrid_brain")


def test_decay_uses_correct_collection():
    assert memory_decay.COLLECTION == "second_brain" or "second_brain" in memory_decay.COLLECTION


def test_high_importance_protected(monkeypatch):
    old_date = (datetime.now() - timedelta(days=200)).isoformat()
    long_text = "John Doe achieved 23% growth in Jan 2026. " * 8
    point = SimpleNamespace(
        id=1,
        payload={
            "importance": 95,
            "source": "manual_commit",
            "date": old_date,
            "text": long_text,
            "retrieval_count": 10,
            "connected_to": ["A", "B", "C", "D"],
        },
    )

    class FakeQdrant:
        def __init__(self):
            self.calls = 0

        def scroll(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [point], None
            return [], None

    monkeypatch.setattr(memory_decay, "qdrant", FakeQdrant())
    stats, archive_cands, softdel_cands, _ = memory_decay.scan_memories()
    assert stats["protected_high_importance"] == 1
    assert archive_cands == []
    assert softdel_cands == []


def test_low_importance_archived(monkeypatch):
    old_date = (datetime.now() - timedelta(days=100)).isoformat()
    point = SimpleNamespace(
        id=2,
        payload={
            "importance": 20,
            "source": "web_page",
            "date": old_date,
            "text": "short note",
            "retrieval_count": 0,
        },
    )

    class FakeQdrant:
        def __init__(self):
            self.calls = 0

        def scroll(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [point], None
            return [], None

    monkeypatch.setattr(memory_decay, "qdrant", FakeQdrant())
    stats, archive_cands, softdel_cands, _ = memory_decay.scan_memories()
    assert stats["archive_candidates"] == 1
    assert len(archive_cands) == 1
    assert softdel_cands == []


def test_temporal_decay_math():
    now = datetime.now(timezone.utc)
    date = (now - timedelta(days=14)).isoformat()
    rows = [{"score": 1.0, "date": date, "importance": 20, "retrieval_count": 0}]
    out = hybrid_brain.apply_temporal_decay(rows)

    half_life = 14
    stability = half_life / math.log(2)
    decay_factor = math.exp(-14 / stability)
    expected = round(1.0 * (0.2 + 0.8 * decay_factor), 4)

    assert out[0]["effective_half_life"] == 14
    assert abs(out[0]["score"] - expected) <= 0.01
