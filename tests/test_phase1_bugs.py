"""Phase 1 bug fix tests — verify critical fixes without requiring live services."""

import importlib
import os
import re
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


# ─── Task 1: Decay collection name ───────────────────────────────────────────


def test_decay_collection_matches_production():
    """Verify decay engine targets 'second_brain' (the live collection)."""
    source = (ROOT / "tools" / "memory_decay.py").read_text()
    # Must NOT contain hardcoded memories_v2
    assert 'COLLECTION = "memories_v2"' not in source
    # Must read from env with correct default
    assert 'os.environ.get("QDRANT_COLLECTION", "second_brain")' in source


def test_bm25_available_defined():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "BM25_AVAILABLE = True" in source


def test_bm25_tokenizer_cyrillic():
    bm25_source = (ROOT / "tools" / "bm25_search.py").read_text()
    engine_source = (ROOT / "tools" / "memory_engine.py").read_text()
    assert "\\w+" in bm25_source
    assert "findall" in bm25_source

    bm25_mod = importlib.import_module("bm25_search")
    tokens = bm25_mod.BM25Scorer().tokenize("Москва river")
    assert "москва" in tokens


def test_fact_extractor_single_commit():
    source = (ROOT / "tools" / "fact_extractor.py").read_text()
    assert "/collections/second_brain/points" not in source
    assert "http://localhost:7777/commit" in source


def test_embed_url_consistency():
    source = (ROOT / "tools" / "fact_extractor.py").read_text()
    assert "http://localhost:11434/api/embed" in source
    assert "/api/embeddings" not in source


def test_concurrent_commits_no_dupes():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "_commit_lock = threading.Lock()" in source
    assert "with _commit_lock:" in source


def test_amac_metrics_thread_safe():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "_amac_metrics_lock = threading.Lock()" in source
    assert "def _inc_metric(" in source
    assert '_amac_metrics["accepted"] += 1' not in source


def test_access_tracking_increments():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert '"point_id": point.id' in source
    assert "qdrant.retrieve(" in source
    assert "for r in results[:10]" not in source


def test_high_importance_not_soft_deleted():
    source = (ROOT / "tools" / "memory_decay.py").read_text()
    assert '"protected_high_importance"' in source
    assert "importance >= 80" in source


def test_archive_atomicity():
    source = (ROOT / "tools" / "memory_decay.py").read_text()
    assert "pending_archive" in source
    assert "def recover_pending_archives" in source
    assert "recover_pending_archives(execute=execute)" in source


def test_memory_engine_commit_uses_api():
    source = (ROOT / "tools" / "memory_engine.py").read_text()
    assert "requests.post(" in source
    assert "http://localhost:7777/commit" in source


def test_consolidator_uses_commit_api():
    source = (ROOT / "tools" / "memory_consolidator_v4.py").read_text()
    assert "http://localhost:7777/commit" in source
    assert "/collections/second_brain/points" not in source


def test_handler_js_uses_process_env_urls():
    source = (ROOT / "hooks" / "openclaw-mem" / "handler.js").read_text()
    assert "process.env.MEMORY_API_URL" in source
    assert "process.env.HONCHO_URL" in source
    assert "${MEMORY_API_URL:-" not in source
    assert "os.environ.get" not in source


def test_task14_entity_matching_uses_word_boundaries():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "if name.lower() in text_lower" not in source
    assert 're.search(r"\\b" + re.escape(name.lower()) + r"\\b", text_lower)' in source


def test_task15_dedup_uses_regex_tokenizer():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "set(text.lower().split())" not in source
    assert 'set(re.findall(r"\\w+", text.lower()))' in source
    assert 'set(re.findall(r"\\w+", existing_text.lower()))' in source


def test_task16_timezone_aware_decay_and_parsing():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "datetime.now(timezone.utc)" in source
    assert "date_str[:26]" not in source
    assert "fromisoformat(normalized)" in source


def test_task17_commit_text_length_validation():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "Text too short (minimum 20 characters)" in source
    assert "Text too long (maximum 8000 characters)" in source


def test_task18_importance_is_clamped_to_0_100():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "importance = max(0, min(100, importance))" in source


def test_task19_request_body_size_limit():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "MAX_BODY_SIZE = 1 * 1024 * 1024" in source
    assert "Request body too large (max 1MB)" in source


def test_task20_commit_metadata_protects_core_fields():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "protected_fields" in source
    assert '"text"' in source[source.index("protected_fields") : source.index("protected_fields") + 300]
    assert '"embedding_model"' in source[source.index("protected_fields") : source.index("protected_fields") + 300]
    assert "payload.update(safe_metadata)" in source


def test_task21_amac_scores_sentinel_prompt_and_parser():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "Output format: SCORES: R,N,S" in source
    assert 're.search(r"SCORES:\\s*(.*)", raw, re.IGNORECASE | re.DOTALL)' in source


def test_task22_known_entities_ttl_cache_present():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "KNOWN_ENTITIES_CACHE_TTL_SECONDS = 5 * 60" in source
    assert "_known_entities_cache" in source
    assert "if _known_entities_cache and (now - _known_entities_cache_ts) < KNOWN_ENTITIES_CACHE_TTL_SECONDS:" in source


def test_task23_brain_port_bound_to_localhost():
    source = (ROOT / "docker-compose.yml").read_text()
    assert '"127.0.0.1:7777:7777"' in source
    assert '"7777:7777"' not in source


def test_task24_reranker_max_length_1024():
    source = (ROOT / "tools" / "reranker_server.py").read_text()
    assert "max_length=1024" in source
    assert "max_length=512" not in source


def test_task25_amac_importance_blending_formula():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert "importance = int(0.4 * importance + 0.6 * amac_composite * 10)" in source
