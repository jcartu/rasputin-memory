#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_PARSE_ERROR = 2

RUNTIME_LINE_RE = re.compile(
    r"^\s*✅\s+(conv-\d+):\s+([0-9]+(?:\.[0-9]+)?)% \(excl\. adversarial, (\d+) Qs\)\s*$"
)
REPORT_LINE_RE = re.compile(
    r"^- \*\*(conv-\d+)\*\*: ([0-9]+(?:\.[0-9]+)?)% \((\d+)/(\d+) excl\. adv\)\s*$"
)
ARTIFACT_COMMIT_RE = re.compile(r"^([0-9a-f]{40})-(.+)-(?:production|compare)\.json$")


@dataclass(frozen=True)
class ArtifactStats:
    conv_id: str
    correct: int
    total: int
    pct: float
    prediction_hash: str


@dataclass(frozen=True)
class LogStats:
    conv_id: str
    pct: float
    total: int | None
    correct: int | None
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify benchmark artifact per-conv scores against a run log.")
    parser.add_argument("--artifact", required=True, help="Path to benchmark result JSON")
    parser.add_argument("--log", required=True, help="Path to benchmark run log")
    parser.add_argument("--tolerance-pp", type=float, default=0.5, help="Allowed percentage-point drift (default: 0.5)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant base URL")
    return parser.parse_args()


def extract_rows(data: object) -> list[dict]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []

    rows = data.get("results") or data.get("rows") or data.get("data") or []
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(rows, dict):
        flat: list[dict] = []
        for value in rows.values():
            if isinstance(value, list):
                flat.extend(row for row in value if isinstance(row, dict))
        return flat
    return []


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return extract_rows(json.load(f))


def conv_id_for(row: dict) -> str:
    return str(row.get("conv_id") or row.get("conversation_id") or row.get("sample_id") or "?")


def prediction_text(row: dict) -> str:
    return str(row.get("predicted") or row.get("predicted_answer") or row.get("answer") or row.get("prediction") or "")


def is_non_adversarial(row: dict) -> bool:
    category = row.get("category")
    if category is not None:
        return int(category) != 5
    cat_name = str(row.get("cat_name", ""))
    return "adv" not in cat_name.lower()


def compute_artifact_stats(rows: list[dict]) -> dict[str, ArtifactStats]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not is_non_adversarial(row):
            continue
        grouped.setdefault(conv_id_for(row), []).append(row)

    stats: dict[str, ArtifactStats] = {}
    for conv_id, conv_rows in sorted(grouped.items()):
        correct = sum(1 for row in conv_rows if row.get("correct"))
        total = len(conv_rows)
        payload = []
        for row in conv_rows:
            payload.append(
                {
                    "qi": row.get("qi"),
                    "question": row.get("question", ""),
                    "prediction": prediction_text(row),
                }
            )
        payload.sort(key=lambda item: (str(item["question"]), str(item["qi"]), str(item["prediction"])))
        hasher = hashlib.sha256()
        for item in payload:
            hasher.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            hasher.update(b"\n")
        stats[conv_id] = ArtifactStats(
            conv_id=conv_id,
            correct=correct,
            total=total,
            pct=(correct / total * 100.0) if total else 0.0,
            prediction_hash=hasher.hexdigest(),
        )
    return stats


def parse_log_stats(path: Path) -> dict[str, LogStats]:
    parsed: dict[str, LogStats] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        report_match = REPORT_LINE_RE.match(line)
        if report_match:
            conv_id, pct, correct, total = report_match.groups()
            parsed[conv_id] = LogStats(conv_id, float(pct), int(total), int(correct), "report")
            continue

        runtime_match = RUNTIME_LINE_RE.match(line)
        if runtime_match:
            conv_id, pct, total = runtime_match.groups()
            if conv_id not in parsed or parsed[conv_id].source != "report":
                parsed[conv_id] = LogStats(conv_id, float(pct), int(total), None, "runtime")
    return parsed


def artifact_commit_sha(path: Path) -> str | None:
    match = ARTIFACT_COMMIT_RE.match(path.name)
    if match is None:
        return None
    return match.group(1)


def verify_collection_commit_coupling(rows: list[dict], artifact_path: Path, qdrant_url: str) -> int:
    artifact_sha = artifact_commit_sha(artifact_path)
    if artifact_sha is None:
        print(
            f"PARSE ERROR: could not infer artifact commit SHA from filename {artifact_path.name}",
            file=sys.stderr,
        )
        return EXIT_PARSE_ERROR

    client = QdrantClient(url=qdrant_url)
    conv_ids = sorted({conv_id_for(row) for row in rows})
    exit_code = EXIT_OK
    for conv_id in conv_ids:
        collection = f"locomo_lb_{conv_id.replace('-', '_')}"
        points, _ = client.scroll(collection_name=collection, limit=1, with_payload=True, with_vectors=False)
        if not points:
            print(f"MISMATCH {collection}: collection empty or missing sample payload", file=sys.stderr)
            exit_code = EXIT_MISMATCH
            continue
        payload = points[0].payload or {}
        ingest_sha = payload.get("_ingest_commit_sha")
        if ingest_sha != artifact_sha:
            print(
                f"MISMATCH {collection}: artifact commit {artifact_sha} != collection ingest {ingest_sha}",
                file=sys.stderr,
            )
            exit_code = EXIT_MISMATCH
            continue
        print(f"COUPLED {collection}: artifact commit matches ingest SHA {artifact_sha[:12]}")
    return exit_code


def compare_stats(artifact: dict[str, ArtifactStats], log_stats: dict[str, LogStats], tolerance_pp: float) -> int:
    if not log_stats:
        print("PARSE ERROR: no per-conversation score lines found in log", file=sys.stderr)
        return EXIT_PARSE_ERROR

    if set(artifact) != set(log_stats):
        missing = sorted(set(artifact) - set(log_stats))
        extra = sorted(set(log_stats) - set(artifact))
        print("PARSE ERROR: artifact/log conversation sets differ", file=sys.stderr)
        if missing:
            print(f"  in artifact, missing from log: {', '.join(missing)}", file=sys.stderr)
        if extra:
            print(f"  in log, missing from artifact: {', '.join(extra)}", file=sys.stderr)
        return EXIT_PARSE_ERROR

    exit_code = EXIT_OK
    for conv_id in sorted(artifact):
        artifact_stats = artifact[conv_id]
        log_row = log_stats[conv_id]
        pct_ok = abs(artifact_stats.pct - log_row.pct) <= tolerance_pp
        total_ok = log_row.total is None or artifact_stats.total == log_row.total
        correct_ok = log_row.correct is None or artifact_stats.correct == log_row.correct
        status = "MATCH" if pct_ok and total_ok and correct_ok else "MISMATCH"
        count_text = "" if log_row.correct is None else f"({log_row.correct}/{log_row.total}) "
        print(
            f"{status} {conv_id}: artifact={artifact_stats.pct:.2f}% ({artifact_stats.correct}/{artifact_stats.total}) "
            f"log={log_row.pct:.2f}% {count_text}source={log_row.source} hash={artifact_stats.prediction_hash[:16]}"
        )
        if status != "MATCH":
            exit_code = EXIT_MISMATCH

    overall_correct = sum(s.correct for s in artifact.values())
    overall_total = sum(s.total for s in artifact.values())
    overall_pct = 100.0 * overall_correct / max(overall_total, 1)
    print(
        f"\nOVERALL non-adv: {overall_correct}/{overall_total} = {overall_pct:.2f}% "
        f"({'MATCH' if exit_code == EXIT_OK else 'MISMATCH'})"
    )
    return exit_code


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact)
    rows = load_rows(artifact_path)
    if not rows:
        print("PARSE ERROR: artifact did not contain any benchmark rows", file=sys.stderr)
        return EXIT_PARSE_ERROR

    artifact_stats = compute_artifact_stats(rows)
    if not artifact_stats:
        print("PARSE ERROR: artifact did not contain any non-adversarial rows", file=sys.stderr)
        return EXIT_PARSE_ERROR

    log_stats = parse_log_stats(Path(args.log))
    stats_exit = compare_stats(artifact_stats, log_stats, args.tolerance_pp)
    if stats_exit == EXIT_PARSE_ERROR:
        return stats_exit
    coupling_exit = verify_collection_commit_coupling(rows, artifact_path, args.qdrant_url)
    if coupling_exit == EXIT_PARSE_ERROR:
        return coupling_exit
    if stats_exit == EXIT_MISMATCH or coupling_exit == EXIT_MISMATCH:
        return EXIT_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
