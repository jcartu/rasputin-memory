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
