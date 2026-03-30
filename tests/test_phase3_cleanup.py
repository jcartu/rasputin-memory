import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock
import re

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import config  # noqa: E402
import bm25_search  # noqa: E402
import backfill_schema  # noqa: E402
import hybrid_brain  # noqa: E402

load_config: Any = getattr(config, "load_config")


def test_config_loads(tmp_path):
    config_path = tmp_path / "rasputin.toml"
    config_path.write_text(
        """
[server]
port = 7777
host = "127.0.0.1"

[qdrant]
url = "http://localhost:6333"
collection = "second_brain"

[graph]
host = "localhost"
port = 6380
graph_name = "brain"
disabled = false

[embeddings]
url = "http://localhost:11434/api/embed"
model = "nomic-embed-text"
prefix_query = "search_query: "
prefix_doc = "search_document: "

[reranker]
url = "http://localhost:8006/rerank"
timeout = 15
enabled = true

[amac]
threshold = 4.0
timeout = 30
model = "qwen2.5:14b"

[scoring]
decay_half_life_low = 14
decay_half_life_medium = 60
decay_half_life_high = 365

[entities]
known_entities_path = "config/known_entities.json"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg["qdrant"]["collection"] == "second_brain"
    assert cfg["graph"]["graph_name"] == "brain"
    assert cfg["embeddings"]["model"] == "nomic-embed-text"


def test_config_env_override(monkeypatch, tmp_path):
    config_path = tmp_path / "rasputin.toml"
    config_path.write_text(
        """
[server]
port = 7777
host = "127.0.0.1"

[qdrant]
url = "http://localhost:6333"
collection = "second_brain"

[graph]
host = "localhost"
port = 6380
graph_name = "brain"
disabled = false

[embeddings]
url = "http://localhost:11434/api/embed"
model = "nomic-embed-text"
prefix_query = "search_query: "
prefix_doc = "search_document: "

[reranker]
url = "http://localhost:8006/rerank"
timeout = 15
enabled = true

[amac]
threshold = 4.0
timeout = 30
model = "qwen2.5:14b"

[scoring]
decay_half_life_low = 14
decay_half_life_medium = 60
decay_half_life_high = 365

[entities]
known_entities_path = "config/known_entities.json"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("QDRANT_COLLECTION", "memories_v2")
    monkeypatch.setenv("FALKORDB_GRAPH", "brain_prod")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text-v2")

    cfg = load_config(str(config_path))

    assert cfg["qdrant"]["collection"] == "memories_v2"
    assert cfg["graph"]["graph_name"] == "brain_prod"
    assert cfg["embeddings"]["model"] == "nomic-embed-text-v2"


def test_commit_includes_schema_version(monkeypatch):
    mock_qdrant = Mock()
    monkeypatch.setattr(hybrid_brain, "qdrant", mock_qdrant)
    monkeypatch.setattr(hybrid_brain, "check_duplicate", lambda *_args, **_kwargs: (False, None, 0))
    monkeypatch.setattr(hybrid_brain, "get_embedding", lambda *_args, **_kwargs: [0.1] * 768)
    monkeypatch.setattr(hybrid_brain, "extract_entities_fast", lambda _text: [])
    monkeypatch.setattr(hybrid_brain, "check_contradictions", lambda *_a, **_k: [])

    result = hybrid_brain.commit_memory("A" * 64, source="conversation", importance=60)

    assert result["ok"] is True
    upsert_kwargs = mock_qdrant.upsert.call_args.kwargs
    payload = upsert_kwargs["points"][0].payload
    assert payload["embedding_model"] == hybrid_brain.EMBED_MODEL
    assert payload["schema_version"] == "0.3"


def test_backfill_schema_sets_payload(monkeypatch):
    class Point:
        def __init__(self, point_id):
            self.id = point_id

    class FakeQdrant:
        def __init__(self):
            self.calls = 0
            self.payload_calls = []

        def scroll(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [Point(1), Point(2)], "next"
            if self.calls == 2:
                return [Point(3)], None
            return [], None

        def set_payload(self, **kwargs):
            self.payload_calls.append(kwargs)

    fake = FakeQdrant()
    monkeypatch.setattr(backfill_schema, "QdrantClient", lambda url: fake)
    monkeypatch.setattr(
        backfill_schema,
        "load_config",
        lambda: {
            "qdrant": {"url": "http://localhost:6333", "collection": "second_brain"},
            "embeddings": {"model": "nomic-embed-text"},
        },
    )

    updated = backfill_schema.backfill_schema(batch_size=2)

    assert updated == 3
    assert len(fake.payload_calls) == 2
    assert fake.payload_calls[0]["payload"]["schema_version"] == "0.3"


def test_no_dead_imports():
    removed_paths = [
        "tools/hybrid_brain_v2_tenant.py",
        "tools/memory_consolidate.py",
        "tools/smart_memory_query.py",
        "graph-brain/graph_query.py",
        "graph-brain/migrate_to_graph.py",
        "brainbox/brainbox.py",
        "predictive-memory/access_tracker.py",
        "storm-wiki/generate.py",
        "honcho/honcho-query.sh",
    ]
    for relative in removed_paths:
        assert not (ROOT / relative).exists()


def test_server_handles_unexpected_error(monkeypatch):
    sent = {}

    class DummyHandler:
        def _handle_get(self):
            raise RuntimeError("boom")

        def _send_json(self, payload, status=200):
            sent["payload"] = payload
            sent["status"] = status

    hybrid_brain.HybridHandler.do_GET(DummyHandler())

    assert sent["status"] == 500
    assert sent["payload"] == {"error": "Internal server error"}


def test_all_files_use_config_collection():
    collection_assignment = re.compile(r"COLLECTION\s*=\s*\"second_brain\"")
    hardcoded_collection_url = re.compile(r"/collections/second_brain")

    for py_file in (ROOT / "tools").glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert collection_assignment.search(text) is None, f"Hardcoded collection assignment in {py_file}"
        assert hardcoded_collection_url.search(text) is None, f"Hardcoded collection URL in {py_file}"


def test_rrf_handles_length_mismatch():
    dense = [
        {"score": 0.9, "payload": {}},
        {"score": 0.8, "payload": {}},
        {"score": 0.7, "payload": {}},
    ]
    bm25 = [2.0]

    fused = bm25_search.reciprocal_rank_fusion(dense, bm25)

    assert len(fused) == 3
    assert all("bm25_score" in item for item in fused)
