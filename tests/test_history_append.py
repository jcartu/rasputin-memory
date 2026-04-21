from __future__ import annotations

import csv
from pathlib import Path

from benchmarks import bench_runner
from scripts import verify_bench_artifact as verify


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def good_artifact() -> dict[str, object]:
    return {
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "commit_short": "01234567",
        "date": "2026-04-21T12:34:56",
        "benchmark": "locomo",
        "mode": "production",
        "answer_model": "claude-haiku-4-5-20251001",
        "judge_model": "gpt-4o-mini-2024-07-18",
        "context_chunks": 60,
        "search_limit": 60,
        "note": "baseline",
        "fact_count": 12,
        "extractor_provider": "openai",
        "wall_clock_seconds": 123.4,
        "results": [
            {
                "conv_id": "conv-1",
                "qi": 1,
                "question": "Q",
                "predicted": "A",
                "category": 1,
                "cat_name": "open-domain",
                "correct": True,
            }
        ],
    }


def corrupt_artifact() -> dict[str, object]:
    return {
        "commit": "fedcba9876543210fedcba9876543210fedcba98",
        "commit_short": "fedcba98",
        "benchmark": "locomo",
        "mode": "production",
    }


def test_append_history_row_writes_score_and_metadata(tmp_path: Path) -> None:
    artifact = good_artifact()
    artifact_path = tmp_path / "0123456789abcdef0123456789abcdef01234567-locomo-production.json"
    history_path = tmp_path / "history.csv"

    rows = verify.extract_rows(artifact)
    stats = verify.compute_artifact_stats(rows)
    verify.append_history_row(artifact, rows, verify.overall_score_for(stats), artifact_path, history_path)

    written_rows = read_csv_rows(history_path)
    assert len(written_rows) == 1
    row = written_rows[0]
    assert row["commit"] == artifact["commit"]
    assert row["headline_score"] == "100.00"
    assert row["answer_model"] == artifact["answer_model"]
    assert row["judge_model"] == artifact["judge_model"]
    assert row["note"] == "baseline | categories:open-domain=1/1 | facts=12 | extractor=openai | wall_clock=123.4"


def test_append_history_row_writes_error_for_corrupt_artifact(tmp_path: Path) -> None:
    artifact = corrupt_artifact()
    artifact_path = tmp_path / "fedcba9876543210fedcba9876543210fedcba98-locomo-production.json"
    history_path = tmp_path / "history.csv"

    verify.append_history_row(artifact, [], "ERROR", artifact_path, history_path)

    written_rows = read_csv_rows(history_path)
    assert len(written_rows) == 1
    assert written_rows[0]["headline_score"] == "ERROR"


def test_append_history_row_is_idempotent_for_same_commit(tmp_path: Path) -> None:
    artifact = good_artifact()
    artifact_path = tmp_path / "0123456789abcdef0123456789abcdef01234567-locomo-production.json"
    history_path = tmp_path / "history.csv"

    rows = verify.extract_rows(artifact)
    stats = verify.compute_artifact_stats(rows)
    score = verify.overall_score_for(stats)
    verify.append_history_row(artifact, rows, score, artifact_path, history_path)
    verify.append_history_row(artifact, rows, score, artifact_path, history_path)

    assert len(read_csv_rows(history_path)) == 1


def test_repo_is_dirty_ignores_history_csv_only(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], capture_output: bool, cwd: str) -> object:
        captured["args"] = args

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(bench_runner.subprocess, "run", fake_run)

    assert bench_runner.repo_is_dirty() is False
    assert ":(exclude)benchmarks/results/history.csv" in captured["args"]


def test_repo_is_dirty_reports_other_changes(monkeypatch) -> None:
    def fake_run(args: list[str], capture_output: bool, cwd: str) -> object:
        assert ":(exclude)benchmarks/results/history.csv" in args

        class Result:
            returncode = 1

        return Result()

    monkeypatch.setattr(bench_runner.subprocess, "run", fake_run)

    assert bench_runner.repo_is_dirty() is True
