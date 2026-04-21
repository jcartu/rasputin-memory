"""Per-commit ingest cache for locomo bench collections.

When a Qdrant collection was ingested on the same git HEAD SHA the search-only
run can reuse it without re-ingesting. The cache is an auxiliary fast-path on
top of the S1.2 ``_ingest_commit_sha`` payload coupling: the cache status file
simply mirrors what's already stamped on Qdrant points, so we can check a
single filesystem read instead of scrolling every collection.

Cache file schema (``/tmp/bench_runs/ingest_cache_status.json`` by default)::

    {
      "collections": {
        "locomo_lb_conv_26": {
          "commit_sha": "<40-char git HEAD at ingest>",
          "point_count": 1311,
          "ingest_timestamp": "<ISO 8601 UTC>",
          "extractor_provider": "cerebras",
          "embedder": "nomic-embed-text"
        },
        ...
      }
    }

Override path via ``BENCH_CACHE_STATUS_PATH`` env var for tests.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CACHE_STATUS_PATH = Path(
    os.environ.get("BENCH_CACHE_STATUS_PATH", "/tmp/bench_runs/ingest_cache_status.json")
)


def get_current_sha(repo_root: Path) -> str:
    """Return current git HEAD SHA. Caller handles subprocess errors."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        cwd=str(repo_root),
        text=True,
    )
    return result.stdout.strip()


def load_cache_status(path: Path | None = None) -> dict[str, Any]:
    target = path or DEFAULT_CACHE_STATUS_PATH
    if not target.exists():
        return {"collections": {}}
    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"collections": {}}
    if not isinstance(data, dict):
        return {"collections": {}}
    data.setdefault("collections", {})
    if not isinstance(data["collections"], dict):
        data["collections"] = {}
    return data


def save_cache_status(status: dict[str, Any], path: Path | None = None) -> None:
    target = path or DEFAULT_CACHE_STATUS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")


def update_cache_entry(
    collection: str,
    *,
    commit_sha: str,
    point_count: int,
    extractor_provider: str,
    embedder: str,
    path: Path | None = None,
) -> None:
    status = load_cache_status(path)
    status["collections"][collection] = {
        "commit_sha": commit_sha,
        "point_count": int(point_count),
        "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
        "extractor_provider": extractor_provider,
        "embedder": embedder,
    }
    save_cache_status(status, path)


def is_cache_fresh(
    collection: str,
    current_sha: str,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the cache entry iff collection's stored commit_sha matches current_sha."""
    status = load_cache_status(path)
    entry = status.get("collections", {}).get(collection)
    if not isinstance(entry, dict):
        return None
    if entry.get("commit_sha") != current_sha:
        return None
    return entry


def format_cache_info(path: Path | None = None) -> str:
    target = path or DEFAULT_CACHE_STATUS_PATH
    status = load_cache_status(target)
    collections = status.get("collections", {}) or {}
    lines: list[str] = [f"Cache status at {target}"]
    if not collections:
        lines.append("  (empty)")
        return "\n".join(lines)
    for name in sorted(collections):
        entry = collections[name]
        sha = str(entry.get("commit_sha", "?"))[:12]
        pts = entry.get("point_count", "?")
        ts = entry.get("ingest_timestamp", "?")
        ext = entry.get("extractor_provider", "?")
        emb = entry.get("embedder", "?")
        lines.append(f"  {name}: commit={sha} pts={pts} ts={ts} extractor={ext} embed={emb}")
    return "\n".join(lines)
