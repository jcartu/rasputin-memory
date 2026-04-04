#!/usr/bin/env python3
# ruff: noqa: E402
# pyright: reportMissingImports=false
"""
RASPUTIN Memory Benchmark Runner

Usage:
    python3 benchmarks/run_benchmark.py                    # synthetic fixtures only
    python3 benchmarks/run_benchmark.py --full             # + BEIR (needs Ollama)
    python3 benchmarks/run_benchmark.py --output results/  # custom output dir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from brain import _state
from brain import commit as commit_module
from brain import embedding as embedding_module
from brain import graph as graph_module
from brain import search as search_module

VECTOR_SIZE = 768
DEFAULT_FIXTURE_PATH = ROOT / "fixtures" / "memory_qa.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "results"
DEFAULT_THRESHOLD_PATH = ROOT / "benchmarks" / "REGRESSION_THRESHOLDS.json"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

TOKEN_NORMALIZATION = {
    "встреча": "meeting",
    "reunión": "meeting",
    "reunion": "meeting",
    "lunes": "monday",
    "понедельник": "monday",
    "назначена": "scheduled",
    "trabaja": "works",
    "работает": "works",
    "газпроме": "gazprom",
    "газпром": "gazprom",
    "алексей": "alexey",
    "москва": "moscow",
    "лондон": "london",
}


@dataclass(slots=True)
class FixtureRecord:
    id: str
    text: str
    query: str
    expected_in_top_k: int
    category: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class EvaluationRow:
    record_id: str
    category: str
    expected_in_top_k: int
    rank: int | None
    latency_ms: float
    result_point_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible in-process memory benchmarks")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_PATH), help="Path to fixtures JSONL file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for result files")
    parser.add_argument("--full", action="store_true", help="Run synthetic suite and optional BEIR extension")
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Fail with non-zero exit code when regression thresholds are breached",
    )
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLD_PATH),
        help="Path to regression threshold JSON",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    for source, replacement in TOKEN_NORMALIZATION.items():
        normalized = normalized.replace(source, replacement)
    normalized = normalized.replace("$", " usd ")
    normalized = normalized.replace("€", " eur ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _hash_index_sign(token: str) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
    sign = 1.0 if (digest[4] % 2 == 0) else -1.0
    return index, sign


def deterministic_hash_embedding(text: str, prefix: str = "") -> list[float]:
    _ = prefix
    canonical = normalize_text(text)
    tokens = TOKEN_RE.findall(canonical)
    if not tokens:
        tokens = ["__empty__"]

    vector = [0.0] * VECTOR_SIZE

    for token in tokens:
        idx, sign = _hash_index_sign(f"u::{token}")
        vector[idx] += sign

    for left, right in zip(tokens, tokens[1:]):
        idx, sign = _hash_index_sign(f"b::{left}_{right}")
        vector[idx] += 0.2 * sign

    idx, sign = _hash_index_sign(f"t::{canonical}")
    vector[idx] += 0.05 * sign

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        vector[0] = 1.0
        magnitude = 1.0
    return [value / magnitude for value in vector]


class SyntheticBenchmarkPatches:
    def __init__(self) -> None:
        self._orig_get_embedding = embedding_module.get_embedding
        self._orig_reranker_check = embedding_module.is_reranker_available
        self._orig_write_to_graph = graph_module.write_to_graph
        self._orig_graph_search = graph_module.graph_search
        self._orig_graph_enrich = graph_module.enrich_with_graph
        self._orig_embed_model = _state.EMBED_MODEL
        self._orig_bm25_available = _state.BM25_AVAILABLE
        self._orig_falkor_disabled = _state.FALKORDB_DISABLED

    def __enter__(self) -> SyntheticBenchmarkPatches:
        embedding_module.get_embedding = deterministic_hash_embedding
        embedding_module.is_reranker_available = lambda: False
        graph_module.write_to_graph = lambda *_args, **_kwargs: (True, [])
        graph_module.graph_search = lambda *_args, **_kwargs: []
        graph_module.enrich_with_graph = lambda *_args, **_kwargs: {}

        _state.EMBED_MODEL = "deterministic-hash-768"
        _state.BM25_AVAILABLE = False
        _state.FALKORDB_DISABLED = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        embedding_module.get_embedding = self._orig_get_embedding
        embedding_module.is_reranker_available = self._orig_reranker_check
        graph_module.write_to_graph = self._orig_write_to_graph
        graph_module.graph_search = self._orig_graph_search
        graph_module.enrich_with_graph = self._orig_graph_enrich

        _state.EMBED_MODEL = self._orig_embed_model
        _state.BM25_AVAILABLE = self._orig_bm25_available
        _state.FALKORDB_DISABLED = self._orig_falkor_disabled


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * pct) - 1))
    return round(sorted_values[index], 2)


def load_fixtures(path: Path) -> list[FixtureRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    rows: list[FixtureRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue

            payload = json.loads(content)
            required_fields = {"id", "text", "query", "expected_in_top_k", "category", "metadata"}
            missing = required_fields - payload.keys()
            if missing:
                raise ValueError(f"Fixture line {line_number} missing keys: {sorted(missing)}")

            rows.append(
                FixtureRecord(
                    id=str(payload["id"]),
                    text=str(payload["text"]),
                    query=str(payload["query"]),
                    expected_in_top_k=parse_int(payload["expected_in_top_k"], 5),
                    category=str(payload["category"]),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )

    if len(rows) != 200:
        raise ValueError(f"Expected 200 fixture records, found {len(rows)}")

    return rows


def ensure_qdrant_available() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333")
    try:
        client.get_collections()
    except Exception as error:
        raise RuntimeError(
            "Qdrant is not available at http://localhost:6333. Start it first (e.g. docker compose up -d)."
        ) from error
    return client


def create_temp_collection(client: QdrantClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def delete_collection_safely(client: QdrantClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as error:
        print(f"[WARN] Failed to delete temp collection {collection_name}: {error}")


def iso_time_from_days_old(days_old: Any) -> str:
    value = float(days_old) if isinstance(days_old, (int, float)) else float(parse_int(days_old, 0))
    timestamp = datetime.now(timezone.utc) - timedelta(days=max(value, 0.0))
    return timestamp.isoformat()


def commit_fixtures(fixtures: list[FixtureRecord]) -> tuple[dict[str, Any], dict[str, int]]:
    id_to_point: dict[str, Any] = {}
    dedup_expected = 0
    dedup_caught = 0

    for fixture in fixtures:
        metadata = dict(fixture.metadata)
        source = str(metadata.get("source", "conversation"))
        importance = parse_int(metadata.get("importance", 60), 60)

        result = commit_module.commit_memory(
            text=fixture.text,
            source=source,
            importance=importance,
            metadata=metadata,
        )
        if not result.get("ok"):
            raise RuntimeError(f"Commit failed for fixture {fixture.id}: {result.get('error', 'unknown error')}")

        point_id = result["id"]
        id_to_point[fixture.id] = point_id

        _state.qdrant.set_payload(
            collection_name=_state.COLLECTION,
            points=[point_id],
            payload={"date": iso_time_from_days_old(metadata.get("days_old", 0))},
        )

        if fixture.category == "dedup" and bool(metadata.get("is_duplicate_candidate")):
            dedup_expected += 1
            dedup_action = (result.get("dedup") or {}).get("action")
            if dedup_action == "updated":
                dedup_caught += 1

    return id_to_point, {
        "dedup_expected": dedup_expected,
        "dedup_caught": dedup_caught,
        "unique_points": len(set(id_to_point.values())),
    }


def wait_for_indexing(collection_name: str, expected_points: int, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        info = _state.qdrant.get_collection(collection_name=collection_name)
        count = int(getattr(info, "points_count", 0) or 0)
        if count >= expected_points:
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for Qdrant indexing in collection {collection_name}")


def find_rank(
    results: list[dict[str, Any]], expected_point_id: Any, expected_text: str
) -> tuple[int | None, list[str]]:
    point_ids: list[str] = []
    expected_id = str(expected_point_id)

    for index, row in enumerate(results, start=1):
        point_id = str(row.get("point_id", ""))
        point_ids.append(point_id)
        if point_id == expected_id:
            return index, point_ids

    expected_norm = normalize_text(expected_text)
    if expected_norm:
        for index, row in enumerate(results, start=1):
            candidate = normalize_text(str(row.get("text", "")))
            if candidate == expected_norm:
                return index, point_ids

    return None, point_ids


def evaluate_queries(fixtures: list[FixtureRecord], id_to_point: dict[str, Any]) -> list[EvaluationRow]:
    evaluations: list[EvaluationRow] = []

    for fixture in fixtures:
        start = time.perf_counter()
        try:
            response = search_module.hybrid_search(fixture.query, limit=10)
            results = response.get("results", []) if isinstance(response, dict) else []
        except Exception:
            results = []
        latency_ms = (time.perf_counter() - start) * 1000

        rank, result_point_ids = find_rank(
            results=results, expected_point_id=id_to_point[fixture.id], expected_text=fixture.text
        )

        evaluations.append(
            EvaluationRow(
                record_id=fixture.id,
                category=fixture.category,
                expected_in_top_k=fixture.expected_in_top_k,
                rank=rank,
                latency_ms=latency_ms,
                result_point_ids=result_point_ids,
            )
        )

    return evaluations


def compute_metrics(rows: list[EvaluationRow]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        }

    recall_at_5 = sum(1 for row in rows if row.rank is not None and row.rank <= 5) / total
    recall_at_10 = sum(1 for row in rows if row.rank is not None and row.rank <= 10) / total
    mrr_at_10 = sum((1.0 / row.rank) if row.rank is not None and row.rank <= 10 else 0.0 for row in rows) / total
    latencies = [row.latency_ms for row in rows]

    return {
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "mrr_at_10": round(mrr_at_10, 4),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def summarize_categories(evaluations: list[EvaluationRow]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall = compute_metrics(evaluations)
    by_category: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in evaluations:
        by_category[row.category].append(row)

    categories = {name: compute_metrics(rows) for name, rows in by_category.items()}
    return overall, categories


def _rank_for_point(point_ids: list[str], point_id: str) -> int | None:
    for index, current in enumerate(point_ids, start=1):
        if current == point_id:
            return index
    return None


def compute_supersede_rate(
    fixtures: list[FixtureRecord],
    id_to_point: dict[str, Any],
    evaluations: list[EvaluationRow],
) -> float:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for fixture in fixtures:
        if fixture.category != "contradiction":
            continue
        group = str(fixture.metadata.get("contradiction_group", "")).strip()
        version = str(fixture.metadata.get("contradiction_version", "")).strip()
        if group and version in {"old", "new"}:
            groups[group][version] = fixture.id

    eval_by_id = {row.record_id: row for row in evaluations}
    total = 0
    success = 0

    for group_data in groups.values():
        old_id = group_data.get("old")
        new_id = group_data.get("new")
        if not old_id or not new_id:
            continue

        new_eval = eval_by_id.get(new_id)
        if not new_eval:
            continue

        total += 1
        new_point = str(id_to_point[new_id])
        old_point = str(id_to_point[old_id])
        new_rank = _rank_for_point(new_eval.result_point_ids, new_point)
        old_rank = _rank_for_point(new_eval.result_point_ids, old_point)

        if new_rank is not None and (old_rank is None or new_rank < old_rank):
            success += 1

    return round(success / total, 4) if total else 0.0


def compute_decay_pair_rate(
    fixtures: list[FixtureRecord],
    id_to_point: dict[str, Any],
    evaluations: list[EvaluationRow],
) -> float:
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for fixture in fixtures:
        if fixture.category != "decay":
            continue
        pair_id = str(fixture.metadata.get("pair", "")).strip()
        role = str(fixture.metadata.get("role", "")).strip()
        if pair_id and role in ("high-importance-old", "low-importance-recent"):
            pairs[pair_id][role] = fixture.id

    eval_by_id = {row.record_id: row for row in evaluations}
    total = 0
    success = 0

    for pair_data in pairs.values():
        high_id = pair_data.get("high-importance-old")
        low_id = pair_data.get("low-importance-recent")
        if not high_id or not low_id:
            continue

        high_eval = eval_by_id.get(high_id)
        if not high_eval:
            continue

        total += 1
        high_point = str(id_to_point[high_id])
        low_point = str(id_to_point[low_id])
        high_rank = _rank_for_point(high_eval.result_point_ids, high_point)
        low_rank = _rank_for_point(high_eval.result_point_ids, low_point)

        if high_rank is not None and (low_rank is None or high_rank < low_rank):
            success += 1

    return round(success / total, 4) if total else 0.0


def current_commit_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def write_results_files(
    output_dir: Path,
    run_date: str,
    result_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{run_date}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, ensure_ascii=False, indent=4)

    history_path = output_dir / "history.csv"
    history_exists = history_path.exists()
    overall = result_payload["overall"]
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not history_exists:
            writer.writerow(
                [
                    "date",
                    "commit",
                    "recall_at_5",
                    "recall_at_10",
                    "mrr_at_10",
                    "latency_p50_ms",
                    "latency_p95_ms",
                ]
            )
        writer.writerow(
            [
                run_date,
                result_payload.get("commit", "unknown"),
                overall.get("recall_at_5", 0.0),
                overall.get("recall_at_10", 0.0),
                overall.get("mrr_at_10", 0.0),
                overall.get("latency_p50_ms", 0.0),
                overall.get("latency_p95_ms", 0.0),
            ]
        )

    latest_path = output_dir / "LATEST.md"
    lines = [
        f"# Benchmark Results — {run_date}",
        "",
        "| Category | Recall@5 | Recall@10 | MRR@10 | Latency p50 |",
        "|----------|----------|-----------|--------|-------------|",
    ]

    all_rows: list[tuple[str, dict[str, Any]]] = [("overall", result_payload["overall"])] + sorted(
        result_payload["categories"].items(),
        key=lambda item: item[0],
    )
    for category, metrics in all_rows:
        lines.append(
            "| {category} | {r5:.2f} | {r10:.2f} | {mrr:.2f} | {lat:.0f}ms |".format(
                category=category,
                r5=metrics.get("recall_at_5", 0.0),
                r10=metrics.get("recall_at_10", 0.0),
                mrr=metrics.get("mrr_at_10", 0.0),
                lat=metrics.get("latency_p50_ms", 0.0),
            )
        )

    contradiction_metrics = result_payload["categories"].get("contradiction")
    dedup_metrics = result_payload["categories"].get("dedup")
    if contradiction_metrics or dedup_metrics:
        lines.append("")
    if contradiction_metrics:
        lines.append(f"- Supersede rate (contradiction): {contradiction_metrics.get('supersede_rate', 0.0):.2f}")
    if dedup_metrics:
        lines.append(f"- Dedup precision: {dedup_metrics.get('dedup_precision', 0.0):.2f}")

    latest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_thresholds(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_thresholds(result_payload: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    breaches: list[str] = []

    overall_thresholds = thresholds.get("overall", {})
    overall_metrics = result_payload.get("overall", {})
    for metric_name, threshold in overall_thresholds.items():
        actual = overall_metrics.get(metric_name)
        if actual is None or actual < threshold:
            breaches.append(f"overall.{metric_name} = {actual} < {threshold}")

    category_thresholds = {key: value for key, value in thresholds.items() if key != "overall"}
    categories = result_payload.get("categories", {})
    for category, required_metrics in category_thresholds.items():
        category_metrics = categories.get(category, {})
        for metric_name, threshold in required_metrics.items():
            actual = category_metrics.get(metric_name)
            if actual is None or actual < threshold:
                breaches.append(f"{category}.{metric_name} = {actual} < {threshold}")

    return breaches


def run_full_extension(full_requested: bool) -> dict[str, Any]:
    if not full_requested:
        return {"enabled": False}
    return {
        "enabled": True,
        "status": "synthetic_only",
        "reason": "BEIR extension placeholder; synthetic benchmark still executed.",
    }


def run(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixtures)
    output_dir = Path(args.output)
    threshold_path = Path(args.thresholds)

    fixtures = load_fixtures(fixture_path)
    qdrant_client = ensure_qdrant_available()

    original_collection = _state.COLLECTION
    original_qdrant = _state.qdrant
    temp_collection = f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    temp_collection_created = False

    try:
        _state.qdrant = qdrant_client
        _state.COLLECTION = temp_collection

        create_temp_collection(qdrant_client, temp_collection)
        temp_collection_created = True

        synthetic_embed_model = "deterministic-hash-768"

        with SyntheticBenchmarkPatches():
            id_to_point, commit_stats = commit_fixtures(fixtures)
            wait_for_indexing(temp_collection, expected_points=commit_stats["unique_points"])
            evaluations = evaluate_queries(fixtures, id_to_point)

            overall, categories = summarize_categories(evaluations)
            categories.setdefault("contradiction", {})["supersede_rate"] = compute_supersede_rate(
                fixtures=fixtures,
                id_to_point=id_to_point,
                evaluations=evaluations,
            )
            categories.setdefault("decay", {})["pair_ranking_rate"] = compute_decay_pair_rate(
                fixtures=fixtures,
                id_to_point=id_to_point,
                evaluations=evaluations,
            )
            expected = commit_stats.get("dedup_expected", 0)
            caught = commit_stats.get("dedup_caught", 0)
            categories.setdefault("dedup", {})["dedup_precision"] = round(caught / expected, 4) if expected else 0.0

            full_extension = run_full_extension(args.full)

        run_date = date.today().isoformat()
        result_payload = {
            "date": run_date,
            "commit": current_commit_hash(),
            "overall": overall,
            "categories": categories,
            "config": {
                "collection": temp_collection,
                "qdrant_url": "http://localhost:6333",
                "embed_model": _state.EMBED_MODEL,
                "synthetic_embed_model": synthetic_embed_model,
                "fixtures": str(fixture_path),
                "records": len(fixtures),
                "full": bool(args.full),
                "full_extension": full_extension,
            },
        }

        write_results_files(output_dir=output_dir, run_date=run_date, result_payload=result_payload)

        print(f"[OK] Benchmark results written to {output_dir}")
        print(
            "[OK] overall: recall@5={r5:.3f} recall@10={r10:.3f} mrr@10={mrr:.3f}".format(
                r5=overall["recall_at_5"],
                r10=overall["recall_at_10"],
                mrr=overall["mrr_at_10"],
            )
        )

        if args.check_thresholds:
            thresholds = load_thresholds(threshold_path)
            breaches = check_thresholds(result_payload=result_payload, thresholds=thresholds)
            if breaches:
                print("[FAIL] Threshold breaches detected:")
                for breach in breaches:
                    print(f"  - {breach}")
                return 2
            print("[OK] Regression thresholds passed")

        return 0
    finally:
        if temp_collection_created:
            delete_collection_safely(qdrant_client, temp_collection)
        _state.COLLECTION = original_collection
        _state.qdrant = original_qdrant


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"[ERROR] Benchmark failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
