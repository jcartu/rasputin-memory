from __future__ import annotations

import importlib
import logging
import os
import threading

import redis
import requests
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("hybrid_brain")

try:
    _config_module = importlib.import_module("config")
except ModuleNotFoundError:
    _config_module = importlib.import_module("tools.config")
load_config = _config_module.load_config

try:
    from bm25_search import hybrid_rerank as bm25_rerank
except ModuleNotFoundError:
    from tools.bm25_search import hybrid_rerank as bm25_rerank

logger.info("BM25 reranking enabled")
BM25_AVAILABLE = True

CONFIG = load_config()

SERVER_HOST = CONFIG["server"]["host"]
SERVER_PORT = int(CONFIG["server"]["port"])
QDRANT_URL = CONFIG["qdrant"]["url"]
COLLECTION = CONFIG["qdrant"]["collection"]
EMBED_MODEL = CONFIG["embeddings"]["model"]
EMBED_URL = CONFIG["embeddings"]["url"]
EMBED_PREFIX_QUERY = CONFIG["embeddings"]["prefix_query"]
EMBED_PREFIX_DOC = CONFIG["embeddings"]["prefix_doc"]
RERANKER_URL = CONFIG["reranker"]["url"]
RERANKER_TIMEOUT = int(CONFIG["reranker"]["timeout"])
RERANKER_ENABLED = bool(CONFIG["reranker"]["enabled"])
FALKOR_HOST = CONFIG["graph"]["host"]
FALKOR_PORT = int(CONFIG["graph"]["port"])
GRAPH_NAME = CONFIG["graph"]["graph_name"]
KNOWN_ENTITIES_PATH = CONFIG["entities"]["known_entities_path"]

FALKORDB_DISABLED = bool(CONFIG["graph"]["disabled"])
if FALKORDB_DISABLED:
    logger.warning("FalkorDB disabled via config/env")

qdrant = QdrantClient(url=QDRANT_URL)
_commit_lock = threading.Lock()
_redis_pool = redis.ConnectionPool(host=FALKOR_HOST, port=FALKOR_PORT, max_connections=10)


def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool)


AMAC_THRESHOLD = float(CONFIG["amac"]["threshold"])
AMAC_OLLAMA_MODEL = CONFIG["amac"]["model"]
AMAC_LLM_URL = str(CONFIG.get("amac", {}).get("url") or "http://localhost:11434/v1/chat/completions")
AMAC_REJECT_LOG = os.environ.get("AMAC_REJECT_LOG", "/tmp/amac_rejected.log")
AMAC_TIMEOUT = int(CONFIG["amac"]["timeout"])

_amac_metrics_lock = threading.Lock()
_amac_metrics = {
    "accepted": 0,
    "rejected": 0,
    "bypassed": 0,
    "score_sum": 0.0,
    "score_count": 0,
    "timeout_accepts": 0,
}

MEMORY_API_TOKEN = os.environ.get("MEMORY_API_TOKEN", "")
MAX_BODY_SIZE = 1 * 1024 * 1024

__all__ = [
    "AMAC_LLM_URL",
    "AMAC_OLLAMA_MODEL",
    "AMAC_REJECT_LOG",
    "AMAC_THRESHOLD",
    "AMAC_TIMEOUT",
    "BM25_AVAILABLE",
    "COLLECTION",
    "CONFIG",
    "EMBED_MODEL",
    "EMBED_PREFIX_DOC",
    "EMBED_PREFIX_QUERY",
    "EMBED_URL",
    "FALKORDB_DISABLED",
    "FALKOR_HOST",
    "FALKOR_PORT",
    "GRAPH_NAME",
    "KNOWN_ENTITIES_PATH",
    "MAX_BODY_SIZE",
    "MEMORY_API_TOKEN",
    "QDRANT_URL",
    "RERANKER_ENABLED",
    "RERANKER_TIMEOUT",
    "RERANKER_URL",
    "SERVER_HOST",
    "SERVER_PORT",
    "_amac_metrics",
    "_amac_metrics_lock",
    "_commit_lock",
    "_redis_pool",
    "bm25_rerank",
    "get_redis",
    "load_config",
    "logger",
    "qdrant",
    "redis",
    "requests",
]
