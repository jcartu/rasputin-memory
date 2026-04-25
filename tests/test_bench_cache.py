from __future__ import annotations

import json
from pathlib import Path

from benchmarks import ingest_cache


def test_load_cache_status_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "ingest_cache_status.json"
    status = ingest_cache.load_cache_status(missing)
    assert status == {"collections": {}}


def test_load_cache_status_corrupt_json_returns_empty(tmp_path: Path) -> None:
    corrupt = tmp_path / "ingest_cache_status.json"
    corrupt.write_text("{not valid json")
    status = ingest_cache.load_cache_status(corrupt)
    assert status == {"collections": {}}


def test_load_cache_status_wrong_root_type_returns_empty(tmp_path: Path) -> None:
    bad = tmp_path / "ingest_cache_status.json"
    bad.write_text("[1, 2, 3]")
    assert ingest_cache.load_cache_status(bad) == {"collections": {}}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    data = {
        "collections": {
            "locomo_lb_conv_26": {
                "commit_sha": "a" * 40,
                "point_count": 1311,
                "ingest_timestamp": "2026-04-21T12:00:00+00:00",
                "extractor_provider": "cerebras",
                "embedder": "nomic-embed-text",
            }
        }
    }
    ingest_cache.save_cache_status(data, path)
    assert path.exists()
    loaded = ingest_cache.load_cache_status(path)
    assert loaded == data


def test_update_cache_entry_is_idempotent_and_merges(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="a" * 40,
        point_count=1311,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_41",
        commit_sha="a" * 40,
        point_count=1400,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    status = ingest_cache.load_cache_status(path)
    assert set(status["collections"].keys()) == {"locomo_lb_conv_26", "locomo_lb_conv_41"}
    assert status["collections"]["locomo_lb_conv_26"]["point_count"] == 1311
    assert status["collections"]["locomo_lb_conv_41"]["point_count"] == 1400


def test_update_cache_entry_overwrites_same_collection(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="a" * 40,
        point_count=1311,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="b" * 40,
        point_count=1500,
        extractor_provider="local_vllm",
        embedder="qwen3-embedding",
        path=path,
    )
    status = ingest_cache.load_cache_status(path)
    entry = status["collections"]["locomo_lb_conv_26"]
    assert entry["commit_sha"] == "b" * 40
    assert entry["point_count"] == 1500
    assert entry["extractor_provider"] == "local_vllm"


def test_is_cache_fresh_returns_entry_on_match(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="a" * 40,
        point_count=1311,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    entry = ingest_cache.is_cache_fresh("locomo_lb_conv_26", "a" * 40, path=path)
    assert entry is not None
    assert entry["commit_sha"] == "a" * 40
    assert entry["point_count"] == 1311


def test_is_cache_fresh_returns_none_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="a" * 40,
        point_count=1311,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    assert ingest_cache.is_cache_fresh("locomo_lb_conv_26", "b" * 40, path=path) is None


def test_is_cache_fresh_returns_none_for_missing_collection(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.save_cache_status({"collections": {}}, path)
    assert ingest_cache.is_cache_fresh("locomo_lb_conv_99", "a" * 40, path=path) is None


def test_format_cache_info_empty(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    out = ingest_cache.format_cache_info(path)
    assert "empty" in out


def test_format_cache_info_includes_each_collection(tmp_path: Path) -> None:
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="abcdef0123456789" + "0" * 24,
        point_count=1311,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_42",
        commit_sha="fedcba9876543210" + "0" * 24,
        point_count=1400,
        extractor_provider="local_vllm",
        embedder="qwen3-embedding",
        path=path,
    )
    out = ingest_cache.format_cache_info(path)
    assert "locomo_lb_conv_26" in out
    assert "locomo_lb_conv_42" in out
    assert "pts=1311" in out
    assert "pts=1400" in out
    assert "abcdef012345" in out  # first 12 chars of sha


def test_cache_json_is_stable_across_saves(tmp_path: Path) -> None:
    """Sorted keys + indent=2 so history.csv-style diffs are meaningful."""
    path = tmp_path / "ingest_cache_status.json"
    ingest_cache.update_cache_entry(
        "locomo_lb_conv_26",
        commit_sha="a" * 40,
        point_count=1311,
        extractor_provider="cerebras",
        embedder="nomic-embed-text",
        path=path,
    )
    first = path.read_text()
    # Load and re-save unchanged
    loaded = ingest_cache.load_cache_status(path)
    ingest_cache.save_cache_status(loaded, path)
    second = path.read_text()
    # Timestamps get re-written only on update_cache_entry; plain save must be byte-identical.
    assert first == second
    # Must be valid JSON
    assert json.loads(first) == json.loads(second)
