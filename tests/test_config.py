import importlib


config = importlib.import_module("config")


def test_config_loads_from_toml(monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    cfg = config.load_config("config/rasputin.toml")
    assert cfg["qdrant"]["collection"] == "second_brain"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    cfg = config.load_config("config/rasputin.toml")
    assert cfg["qdrant"]["collection"] == "test_collection"
