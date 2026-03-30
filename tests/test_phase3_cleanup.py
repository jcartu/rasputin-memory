import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import config

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
