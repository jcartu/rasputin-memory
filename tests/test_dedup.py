from __future__ import annotations

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace


def test_score_memory_uses_source_importance():
    scoring_constants = importlib.import_module("pipeline.scoring_constants")
    memory_dedup = importlib.import_module("memory_dedup")

    payload_base = {
        "text": "A" * 100,
        "importance": 50,
        "retrieval_count": 0,
    }

    convo_score = memory_dedup.score_memory({**payload_base, "source": "conversation"})
    unknown_score = memory_dedup.score_memory({**payload_base, "source": "unknown_source"})

    expected_delta = int(scoring_constants.SOURCE_IMPORTANCE["conversation"] * 10) - int(0.5 * 10)
    assert (convo_score - unknown_score) == expected_delta


def test_find_duplicates_filters_by_threshold(monkeypatch):
    memory_dedup = importlib.import_module("memory_dedup")

    fake_results = SimpleNamespace(
        points=[
            SimpleNamespace(id=1, score=1.0, payload={"text": "self"}),
            SimpleNamespace(id=2, score=0.965, payload={"text": "very similar text"}),
            SimpleNamespace(id=3, score=0.75, payload={"text": "not similar"}),
        ]
    )

    class FakeQdrant:
        def query_points(self, **_kwargs):
            return fake_results

    monkeypatch.setattr(memory_dedup, "qdrant", FakeQdrant())
    dupes = memory_dedup.find_duplicates_for_point(point_id=1, vector=[0.1, 0.2], threshold=0.92, limit=10)

    assert len(dupes) == 1
    assert dupes[0]["id"] == 2
    assert dupes[0]["score"] == 0.965


def test_soft_delete_marker_sets_pending_delete_flag(monkeypatch):
    memory_dedup = importlib.import_module("memory_dedup")

    captured: dict[str, object] = {}

    class FakeQdrant:
        def set_payload(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(memory_dedup, "qdrant", FakeQdrant())
    memory_dedup.mark_pending_delete([101, 202])

    assert captured["collection_name"] == memory_dedup.COLLECTION
    assert captured["points"] == [101, 202]
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload.get("pending_delete") is True
    assert payload.get("pending_delete_at")


def test_checkpoint_save_and_load(tmp_path, monkeypatch):
    memory_dedup = importlib.import_module("memory_dedup")

    checkpoint = tmp_path / "dedup_checkpoint.json"
    monkeypatch.setattr(memory_dedup, "CHECKPOINT_FILE", str(checkpoint))

    state = {
        "last_offset": 123,
        "scanned": 20,
        "clusters_found": 2,
        "dupes_marked": 5,
        "processed_ids": [1, 2, 3],
    }
    memory_dedup.save_checkpoint(state)

    loaded = memory_dedup.load_checkpoint()
    assert loaded == state


def test_run_dedup_execute_marks_and_deletes(monkeypatch):
    memory_dedup = importlib.import_module("memory_dedup")

    base_payload = {
        "text": "duplicate memory",
        "source": "conversation",
        "importance": 60,
        "date": datetime.now(timezone.utc).isoformat(),
    }

    class FakeQdrant:
        def __init__(self):
            self.deleted_batches: list[list[int]] = []
            self.marked_batches: list[list[int]] = []

        def get_collection(self, _collection):
            return SimpleNamespace(points_count=2)

        def scroll(self, **_kwargs):
            points = [
                SimpleNamespace(id=1, vector=[1.0, 0.0], payload=base_payload),
                SimpleNamespace(id=2, vector=[1.0, 0.0], payload=base_payload),
            ]
            return points, None

        def query_points(self, **kwargs):
            q = kwargs.get("query")
            if q == [1.0, 0.0]:
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(id=1, score=1.0, payload=base_payload),
                        SimpleNamespace(id=2, score=0.97, payload=base_payload),
                    ]
                )
            return SimpleNamespace(points=[])

        def set_payload(self, collection_name=None, points=None, payload=None, **_kwargs):
            if collection_name == memory_dedup.COLLECTION and payload and payload.get("pending_delete"):
                self.marked_batches.append(list(points or []))

        def delete(self, collection_name=None, points_selector=None, **_kwargs):
            if collection_name == memory_dedup.COLLECTION:
                assert points_selector is not None
                self.deleted_batches.append(list(points_selector.points))

    fake_qdrant = FakeQdrant()
    monkeypatch.setattr(memory_dedup, "qdrant", fake_qdrant)
    monkeypatch.setattr(memory_dedup, "save_checkpoint", lambda _state: None)
    monkeypatch.setattr(memory_dedup, "log_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(memory_dedup.os, "remove", lambda *_args, **_kwargs: None)

    state = memory_dedup.run_dedup(threshold=0.92, limit=2, execute=True, resume=False, batch_size=10)

    assert state["clusters_found"] == 1
    assert state["dupes_marked"] == 1
    assert fake_qdrant.marked_batches == [[2]]
    assert fake_qdrant.deleted_batches == [[2]]
