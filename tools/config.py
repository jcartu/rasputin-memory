#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent.parent / candidate


def load_config(path: str = "config/rasputin.toml") -> dict[str, Any]:
    with _resolve_path(path).open("rb") as f:
        config = tomllib.load(f)

    cfg = copy.deepcopy(config)

    cfg["server"]["host"] = os.environ.get("SERVER_HOST", cfg["server"]["host"])
    cfg["server"]["port"] = int(os.environ.get("PORT", str(cfg["server"]["port"])))

    cfg["qdrant"]["url"] = os.environ.get("QDRANT_URL", cfg["qdrant"]["url"])
    cfg["qdrant"]["collection"] = os.environ.get("QDRANT_COLLECTION", cfg["qdrant"]["collection"])

    cfg["graph"]["host"] = os.environ.get("FALKORDB_HOST", cfg["graph"]["host"])
    cfg["graph"]["port"] = int(os.environ.get("FALKORDB_PORT", str(cfg["graph"]["port"])))
    cfg["graph"]["graph_name"] = os.environ.get("FALKORDB_GRAPH", cfg["graph"]["graph_name"])
    cfg["graph"]["disabled"] = _bool_env("DISABLE_FALKORDB", bool(cfg["graph"]["disabled"]))

    cfg.setdefault("graph_store", {})
    cfg["graph_store"]["sqlite_path"] = os.environ.get(
        "GRAPH_STORE_SQLITE_PATH",
        cfg["graph_store"].get("sqlite_path", "data/graph.db"),
    )
    cfg["graph_store"]["enabled"] = _bool_env(
        "GRAPH_STORE_ENABLED",
        bool(cfg["graph_store"].get("enabled", True)),
    )

    cfg["embeddings"]["url"] = os.environ.get("EMBED_URL", cfg["embeddings"]["url"])
    cfg["embeddings"]["model"] = os.environ.get("EMBED_MODEL", cfg["embeddings"]["model"])
    cfg["embeddings"]["prefix_query"] = os.environ.get("EMBED_PREFIX_QUERY", cfg["embeddings"]["prefix_query"])
    cfg["embeddings"]["prefix_doc"] = os.environ.get("EMBED_PREFIX_DOC", cfg["embeddings"]["prefix_doc"])

    cfg["reranker"]["url"] = os.environ.get("RERANKER_URL", cfg["reranker"]["url"])
    cfg["reranker"]["timeout"] = int(os.environ.get("RERANKER_TIMEOUT", str(cfg["reranker"]["timeout"])))
    cfg["reranker"]["enabled"] = _bool_env("RERANKER_ENABLED", bool(cfg["reranker"]["enabled"]))

    cfg["amac"]["threshold"] = float(os.environ.get("AMAC_THRESHOLD", str(cfg["amac"]["threshold"])))
    cfg["amac"]["timeout"] = int(os.environ.get("AMAC_TIMEOUT", str(cfg["amac"]["timeout"])))
    cfg["amac"]["model"] = os.environ.get("AMAC_MODEL", cfg["amac"]["model"])

    cfg["entities"]["known_entities_path"] = os.environ.get(
        "KNOWN_ENTITIES_PATH", cfg["entities"]["known_entities_path"]
    )

    return cfg
