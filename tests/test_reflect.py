"""Tests for tools/brain/reflect.py.

hybrid_search and LLM calls (urllib.request.urlopen) are mocked.
No running services needed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from brain import _state
from brain import reflect as reflect_mod


def _fake_search_result(results: list[dict[str, Any]], elapsed_ms: float = 50.0) -> dict[str, Any]:
    return {"query": "test", "elapsed_ms": elapsed_ms, "results": results, "stats": {}}


def _make_memories(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "text": f"Memory number {i + 1} about topic X",
            "date": f"2026-04-{10 + i:02d}T00:00:00Z",
            "source": "conversation",
            "point_id": 100 + i,
            "final_score": round(0.9 - i * 0.1, 2),
            "score": round(0.8 - i * 0.1, 2),
        }
        for i in range(n)
    ]


def _mock_urlopen_anthropic(answer_text: str) -> MagicMock:
    body = json.dumps({"content": [{"text": answer_text}]}).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = lambda self: self
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _mock_urlopen_ollama(answer_text: str) -> MagicMock:
    body = json.dumps({"choices": [{"message": {"content": answer_text}}]}).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = lambda self: self
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# -- reflect() main path ---------------------------------------------------


def test_reflect_no_results(monkeypatch):
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result([], elapsed_ms=12),
    )
    result = reflect_mod.reflect("anything")
    assert "don't have any relevant memories" in result["answer"]
    assert result["sources"] == []
    assert result["search_elapsed_ms"] == 12
    assert result["reflect_model"] == reflect_mod.REFLECT_MODEL


def test_reflect_with_results_anthropic(monkeypatch):
    memories = _make_memories(3)
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result(memories, elapsed_ms=55),
    )
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    with patch.object(
        reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_anthropic("Topic X is important.")
    ):
        result = reflect_mod.reflect("What about topic X?", limit=10)

    assert result["answer"] == "Topic X is important."
    assert len(result["sources"]) == 3
    assert result["sources"][0]["point_id"] == 100
    assert result["sources"][0]["score"] == 0.9
    assert result["search_elapsed_ms"] == 55
    assert result["total_elapsed_ms"] > 0


def test_reflect_with_results_ollama(monkeypatch):
    memories = _make_memories(2)
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result(memories),
    )
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "ollama")

    with patch.object(reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_ollama("Ollama says hi.")):
        result = reflect_mod.reflect("question")

    assert result["answer"] == "Ollama says hi."
    assert len(result["sources"]) == 2


def test_reflect_passes_search_params(monkeypatch):
    captured = {}

    def fake_search(query, **kwargs):
        captured.update(kwargs)
        captured["query"] = query
        return _fake_search_result([])

    monkeypatch.setattr(reflect_mod.search, "hybrid_search", fake_search)
    reflect_mod.reflect("q", limit=7, source_filter="decision", collection="custom_col")

    assert captured["query"] == "q"
    assert captured["limit"] == 7
    assert captured["source_filter"] == "decision"
    assert captured["collection"] == "custom_col"


def test_reflect_caps_context_at_15(monkeypatch):
    memories = _make_memories(20)
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result(memories),
    )
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    captured_prompt = {}

    def fake_urlopen(req, **kwargs):
        captured_prompt["body"] = json.loads(req.data.decode())
        return _mock_urlopen_anthropic("answer")

    with patch.object(reflect_mod.urllib.request, "urlopen", side_effect=fake_urlopen):
        result = reflect_mod.reflect("q")

    prompt_text = captured_prompt["body"]["messages"][0]["content"]
    assert "[Memory 15]" in prompt_text
    assert "[Memory 16]" not in prompt_text
    assert len(result["sources"]) == 15


# -- _call_llm routing ------------------------------------------------------


def test_call_llm_routes_to_anthropic(monkeypatch):
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    with patch.object(
        reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_anthropic("from anthropic")
    ) as mock:
        result = reflect_mod._call_llm("prompt")
    assert result == "from anthropic"
    call_url = mock.call_args[0][0].full_url
    assert "anthropic.com" in call_url


def test_call_llm_routes_to_ollama(monkeypatch):
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "ollama")

    with patch.object(reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_ollama("from ollama")) as mock:
        result = reflect_mod._call_llm("prompt")
    assert result == "from ollama"
    call_url = mock.call_args[0][0].full_url
    assert "anthropic.com" not in call_url


def test_call_llm_fallback_to_anthropic_when_no_provider(monkeypatch):
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "unknown")

    with patch.object(reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_anthropic("fallback")) as mock:
        result = reflect_mod._call_llm("prompt")
    assert result == "fallback"
    assert "anthropic.com" in mock.call_args[0][0].full_url


def test_call_llm_fallback_to_ollama_when_no_key(monkeypatch):
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "unknown")

    with patch.object(reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_ollama("ollama fallback")):
        result = reflect_mod._call_llm("prompt")
    assert result == "ollama fallback"


# -- LLM error handling -----------------------------------------------------


def test_anthropic_error_returns_message(monkeypatch):
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    with patch.object(reflect_mod.urllib.request, "urlopen", side_effect=TimeoutError("connection timed out")):
        result = reflect_mod._call_anthropic("prompt")
    assert "Reflection failed" in result
    assert "timed out" in result


def test_ollama_error_returns_message(monkeypatch):
    with patch.object(reflect_mod.urllib.request, "urlopen", side_effect=ConnectionRefusedError("refused")):
        result = reflect_mod._call_ollama("prompt")
    assert "Reflection failed" in result
    assert "refused" in result


# -- Prompt construction ----------------------------------------------------


def test_prompt_includes_query_and_memories(monkeypatch):
    memories = [
        {"text": "Alice likes Python", "date": "2026-04-10", "source": "observation", "point_id": 1, "score": 0.9},
        {"text": "Bob prefers Rust", "date": "2026-04-11", "source": "observation", "point_id": 2, "score": 0.8},
    ]
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result(memories),
    )
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    captured_prompt = {}

    def fake_urlopen(req, **kwargs):
        captured_prompt["text"] = json.loads(req.data.decode())["messages"][0]["content"]
        return _mock_urlopen_anthropic("answer")

    with patch.object(reflect_mod.urllib.request, "urlopen", side_effect=fake_urlopen):
        reflect_mod.reflect("What languages do they prefer?")

    prompt = captured_prompt["text"]
    assert "Alice likes Python" in prompt
    assert "Bob prefers Rust" in prompt
    assert "What languages do they prefer?" in prompt
    assert "[Memory 1]" in prompt
    assert "[Memory 2]" in prompt


def test_source_text_truncated_at_200(monkeypatch):
    long_text = "A" * 500
    memories = [{"text": long_text, "date": "", "source": "", "point_id": 1, "score": 0.5}]
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result(memories),
    )
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    with patch.object(reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_anthropic("ok")):
        result = reflect_mod.reflect("q")

    assert len(result["sources"][0]["text"]) == 200


def test_score_uses_final_score_over_score(monkeypatch):
    memories = [{"text": "mem", "date": "", "source": "", "point_id": 1, "final_score": 0.95, "score": 0.5}]
    monkeypatch.setattr(
        reflect_mod.search,
        "hybrid_search",
        lambda *a, **kw: _fake_search_result(memories),
    )
    monkeypatch.setattr(_state, "ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(reflect_mod, "REFLECT_PROVIDER", "anthropic")

    with patch.object(reflect_mod.urllib.request, "urlopen", return_value=_mock_urlopen_anthropic("ok")):
        result = reflect_mod.reflect("q")

    assert result["sources"][0]["score"] == 0.95
