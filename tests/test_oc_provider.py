"""Smoke tests for oc_provider — cloud model adapter."""

import os


def test_oc_provider_imports():
    """Module imports without error."""
    from tools.oc_provider import OCProvider  # noqa: F401


def test_oc_provider_init_no_config():
    """Initialises gracefully with no OC config file."""
    os.environ["OPENCLAW_CONFIG"] = "/nonexistent/path.json"
    try:
        from tools.oc_provider import OCProvider
        p = OCProvider(config_path="/nonexistent/path.json")
        info = p.info()
        assert "embedding" in info
        assert "llm" in info
        assert "reranker" in info
    finally:
        os.environ.pop("OPENCLAW_CONFIG", None)


def test_oc_provider_info_shape():
    """info() returns expected structure."""
    from tools.oc_provider import OCProvider
    p = OCProvider(config_path="/nonexistent/path.json")
    info = p.info()
    assert isinstance(info, dict)
    for key in ("embedding", "llm", "reranker"):
        assert key in info
    # Model fields present
    assert "embedding_model" in info
    assert "llm_model" in info


def test_oc_provider_health_check_shape():
    """health_check() returns dict with expected keys."""
    from tools.oc_provider import OCProvider
    p = OCProvider(config_path="/nonexistent/path.json")
    status = p.health_check()
    assert isinstance(status, dict)
    # Should have at least embeddings and llm keys
    assert "embeddings" in status or "embedding" in status
    assert "llm" in status


def test_oc_provider_env_detection():
    """Detects OpenAI key from environment."""
    from tools.oc_provider import OCProvider
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-testing"
    try:
        p = OCProvider(config_path="/nonexistent/path.json")
        info = p.info()
        assert info["embedding"] == "openai"
        assert info["llm"] == "openai"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def test_oc_provider_fallback_no_keys():
    """Falls back to Ollama when no API keys available."""
    from tools.oc_provider import OCProvider
    # Remove all API keys
    saved = {}
    for key in ("OPENAI_API_KEY", "COHERE_API_KEY"):
        saved[key] = os.environ.pop(key, None)
    try:
        p = OCProvider(config_path="/nonexistent/path.json")
        info = p.info()
        assert info["embedding"] == "ollama"
        assert info["llm"] == "ollama"
        assert info["reranker"] == "local"
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
