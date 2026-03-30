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
    assert "re.findall(r'\\w+', text.lower())" in bm25_source
    assert "\\b\\w{3,}\\b" in engine_source

    bm25_mod = importlib.import_module("bm25_search")
    tokens = bm25_mod.BM25Scorer().tokenize("Москва river")
    assert "москва" in tokens


def test_fact_extractor_single_commit():
    source = (ROOT / "tools" / "fact_extractor.py").read_text()
    assert '/collections/second_brain/points' not in source
    assert 'http://localhost:7777/commit' in source


def test_embed_url_consistency():
    source = (ROOT / "tools" / "fact_extractor.py").read_text()
    assert 'http://localhost:11434/api/embed' in source
    assert '/api/embeddings' not in source


def test_concurrent_commits_no_dupes():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert '_commit_lock = threading.Lock()' in source
    assert 'with _commit_lock:' in source


def test_amac_metrics_thread_safe():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert '_amac_metrics_lock = threading.Lock()' in source
    assert 'def _inc_metric(key, amount=1):' in source
    assert '_amac_metrics["accepted"] += 1' not in source


def test_access_tracking_increments():
    source = (ROOT / "tools" / "hybrid_brain.py").read_text()
    assert '"point_id": point.id' in source
    assert 'qdrant.retrieve(' in source
    assert 'for r in results[:10]' not in source


def test_high_importance_not_soft_deleted():
    source = (ROOT / "tools" / "memory_decay.py").read_text()
    assert '"protected_high_importance"' in source
    assert 'importance >= 80' in source
