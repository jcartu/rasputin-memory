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
