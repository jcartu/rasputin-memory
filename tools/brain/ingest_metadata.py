from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path


safe_import = importlib.import_module("pipeline._imports").safe_import
_config_module = safe_import("config", "tools.config")
load_config = _config_module.load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACT_TYPE_LIST = ["world", "experience", "inference"]
_INGEST_METADATA_CACHE: dict[str, str] | None = None


def _read_version() -> str:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return str(pyproject["project"]["version"])


def _git_head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        cwd=_REPO_ROOT,
        text=True,
    )
    return result.stdout.strip()


def _extractor_model(provider: str) -> str:
    default_model = os.environ.get("FACT_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
    if provider == "cerebras":
        return os.environ.get("CEREBRAS_FACT_MODEL", os.environ.get("FACT_EXTRACTION_MODEL", "qwen-3-235b-a22b-instruct-2507"))
    return default_model


def _config_hash() -> str:
    config = load_config()
    extractor_provider = os.environ.get("FACT_EXTRACTION_PROVIDER", "anthropic")
    payload = {
        "extractor_provider": extractor_provider,
        "extractor_model": _extractor_model(extractor_provider),
        "embed_model": os.environ.get("EMBED_MODEL", config["embeddings"]["model"]),
        "reranker_endpoint": os.environ.get("RERANKER_URL", config["reranker"]["url"]),
        "fact_type_list": list(_FACT_TYPE_LIST),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get_ingest_metadata() -> dict[str, str]:
    global _INGEST_METADATA_CACHE

    if _INGEST_METADATA_CACHE is None:
        _INGEST_METADATA_CACHE = {
            "_ingest_commit_sha": _git_head_sha(),
            "_ingest_config_hash": _config_hash(),
            "_ingest_timestamp": datetime.now(timezone.utc).isoformat(),
            "_ingest_bench_version": _read_version(),
        }
    return dict(_INGEST_METADATA_CACHE)
