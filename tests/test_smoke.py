"""Smoke tests — verify all major modules import without error."""

import importlib
import sys
from pathlib import Path

# Add repo root and subdirs to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "brainbox"))
sys.path.insert(0, str(ROOT / "storm-wiki"))


def _try_import(module_path: str) -> None:
    """Import a .py file as a module, skipping if optional deps missing."""
    path = ROOT / module_path
    assert path.exists(), f"{module_path} not found"
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    assert spec and spec.loader
    _mod = importlib.util.module_from_spec(spec)
    # We just need to confirm it parses and top-level is valid
    # Full execution may need running services, so we compile only
    with open(path) as f:
        compile(f.read(), str(path), "exec")


def test_import_hybrid_brain():
    _try_import("tools/hybrid_brain.py")


def test_import_memory_engine():
    _try_import("tools/memory_engine.py")


def test_import_bm25_search():
    _try_import("tools/bm25_search.py")


def test_import_reranker_server():
    _try_import("tools/reranker_server.py")


def test_import_memory_consolidate():
    _try_import("tools/memory_consolidate.py")


def test_import_memory_dedup():
    _try_import("tools/memory_dedup.py")


def test_import_memory_decay():
    _try_import("tools/memory_decay.py")


def test_import_fact_extractor():
    _try_import("tools/fact_extractor.py")


def test_import_memory_health_check():
    _try_import("tools/memory_health_check.py")


def test_import_brainbox():
    _try_import("brainbox/brainbox.py")


def test_import_memory_mcp_server():
    _try_import("tools/memory_mcp_server.py")


def test_import_embed_server():
    _try_import("tools/embed_server_gpu1.py")


def test_import_enrich():
    _try_import("tools/enrich_second_brain.py")


def test_import_smart_query():
    _try_import("tools/smart_memory_query.py")


def test_import_storm_generate():
    _try_import("storm-wiki/generate.py")
