from __future__ import annotations

import importlib
from types import SimpleNamespace


bench = importlib.import_module("benchmarks.locomo_leaderboard_bench")
backfill = importlib.import_module("scripts.backfill_ingest_metadata")


class FakeQdrantClient:
    def __init__(self, points: list[SimpleNamespace]):
        self._points = points
        self.set_payload_calls: list[dict[str, object]] = []

    def scroll(
        self,
        collection_name=None,
        limit=128,
        offset=None,
        with_payload=True,
        with_vectors=False,
        **_kwargs,
    ):
        del collection_name, with_payload, with_vectors
        start = int(offset or 0)
        chunk = self._points[start : start + limit]
        next_offset = start + limit if start + limit < len(self._points) else None
        return chunk, next_offset

    def set_payload(self, collection_name=None, points=None, payload=None, **_kwargs):
        self.set_payload_calls.append(
            {"collection_name": collection_name, "points": list(points or []), "payload": dict(payload or {})}
        )
        point_ids = set(points or [])
        for point in self._points:
            if point.id in point_ids:
                point.payload.update(payload or {})


def make_collection_info(points_count: int = 1, dim: int = 768) -> dict:
    return {
        "result": {"points_count": points_count, "config": {"params": {"vectors": {"size": dim}}}},
    }


def test_assert_collection_ready_raises_on_sha_mismatch(monkeypatch):
    fake_client = FakeQdrantClient([SimpleNamespace(id=1, payload={"_ingest_commit_sha": "a" * 40})])
    monkeypatch.setattr(bench, "http_json", lambda *args, **kwargs: make_collection_info())
    monkeypatch.setattr(bench, "QdrantClient", lambda url: fake_client)
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="b" * 40 + "\n"),
    )

    try:
        bench._assert_collection_ready("locomo_lb_conv_26")
    except RuntimeError as exc:
        assert "Re-ingest or pass --allow-cross-commit" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on commit mismatch")


def test_assert_collection_ready_warns_and_continues_with_override(monkeypatch, capsys):
    fake_client = FakeQdrantClient([SimpleNamespace(id=1, payload={"_ingest_commit_sha": "a" * 40})])
    monkeypatch.setattr(bench, "http_json", lambda *args, **kwargs: make_collection_info(points_count=12))
    monkeypatch.setattr(bench, "QdrantClient", lambda url: fake_client)
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="b" * 40 + "\n"),
    )

    count = bench._assert_collection_ready("locomo_lb_conv_26", allow_cross_commit=True)
    captured = capsys.readouterr()

    assert count == 12
    assert "--allow-cross-commit was passed" in captured.err


def test_assert_collection_ready_warns_for_legacy_collection(monkeypatch, capsys):
    fake_client = FakeQdrantClient([SimpleNamespace(id=1, payload={"text": "legacy"})])
    monkeypatch.setattr(bench, "http_json", lambda *args, **kwargs: make_collection_info(points_count=7))
    monkeypatch.setattr(bench, "QdrantClient", lambda url: fake_client)
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="b" * 40 + "\n"),
    )

    count = bench._assert_collection_ready("locomo_lb_conv_26")
    captured = capsys.readouterr()

    assert count == 7
    assert "has no _ingest_commit_sha" in captured.err
    assert "scripts/backfill_ingest_metadata.py" in captured.err


def test_backfill_script_writes_missing_metadata_and_is_idempotent(monkeypatch):
    points = [
        SimpleNamespace(id=1, payload={"text": "missing"}),
        SimpleNamespace(id=2, payload={"text": "existing", "_ingest_commit_sha": "z" * 40}),
    ]
    fake_client = FakeQdrantClient(points)
    monkeypatch.setattr(backfill, "QdrantClient", lambda url: fake_client)
    monkeypatch.setattr(
        backfill,
        "get_ingest_metadata",
        lambda: {
            "_ingest_commit_sha": "x" * 40,
            "_ingest_config_hash": "cfg",
            "_ingest_timestamp": "2026-04-21T00:00:00+00:00",
            "_ingest_bench_version": "0.9.1",
        },
    )

    exit_code = backfill.main(["--collection", "locomo_lb_conv_26", "--commit", "a" * 40, "--config-hash", "cfg"])
    assert exit_code == 0
    assert len(fake_client.set_payload_calls) == 1
    assert points[0].payload["_ingest_commit_sha"] == "a" * 40
    assert points[1].payload["_ingest_commit_sha"] == "z" * 40

    exit_code = backfill.main(["--collection", "locomo_lb_conv_26", "--commit", "a" * 40, "--config-hash", "cfg"])
    assert exit_code == 0
    assert len(fake_client.set_payload_calls) == 1


def test_backfill_script_dry_run_does_not_modify_points(monkeypatch, capsys):
    points = [SimpleNamespace(id=1, payload={"text": "missing"})]
    fake_client = FakeQdrantClient(points)
    monkeypatch.setattr(backfill, "QdrantClient", lambda url: fake_client)
    monkeypatch.setattr(
        backfill,
        "get_ingest_metadata",
        lambda: {
            "_ingest_commit_sha": "x" * 40,
            "_ingest_config_hash": "cfg",
            "_ingest_timestamp": "2026-04-21T00:00:00+00:00",
            "_ingest_bench_version": "0.9.1",
        },
    )

    exit_code = backfill.main(["--collection", "locomo_lb_conv_26", "--commit", "a" * 40, "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert fake_client.set_payload_calls == []
    assert "DRY RUN" in captured.out
    assert "_ingest_commit_sha" not in points[0].payload
