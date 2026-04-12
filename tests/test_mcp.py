"""Tests for the MCP server proxy layer (tools/mcp/server.py).

All HTTP calls to the RASPUTIN API are mocked via urllib.request.urlopen.
No running services needed.  FastMCP is shimmed so the optional dependency
is not required to run these tests.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "mcp"))


def _make_response(data: dict[str, Any], status: int = 200) -> MagicMock:
    body = json.dumps(data).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = lambda self: self
    mock.__exit__ = MagicMock(return_value=False)
    mock.status = status
    return mock


class _FakeFastMCP:
    def __init__(self, name: str, **kwargs: Any):
        self.name = name

    def tool(self, fn: Any) -> Any:
        return fn

    def run(self, **kwargs: Any) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setenv("RASPUTIN_URL", "http://test:7777")
    monkeypatch.setenv("RASPUTIN_TOKEN", "")
    monkeypatch.setenv("RASPUTIN_BANK_ID", "")


@pytest.fixture()
def mcp_server():
    fake_mod = MagicMock()
    fake_mod.FastMCP = _FakeFastMCP
    sys.modules["fastmcp"] = fake_mod
    try:
        if "server" in sys.modules:
            del sys.modules["server"]
        return importlib.import_module("server")
    finally:
        sys.modules.pop("fastmcp", None)


# -- memory_store -----------------------------------------------------------


def test_store_success(mcp_server):
    api_resp = _make_response(
        {
            "ok": True,
            "id": 42,
            "dedup": {"action": "created", "similarity": None},
            "graph": {"written": True, "entities": 2, "connected_to": ["Alice", "Bob"]},
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_store("Alice prefers dark mode", source="observation", importance=80)
    assert "Stored" in result
    assert "42" in result
    assert "Alice" in result


def test_store_rejected(mcp_server):
    api_resp = _make_response(
        {
            "ok": False,
            "rejected": True,
            "reason": "amac_below_threshold",
            "scores": {"relevance": 2, "novelty": 1, "specificity": 0, "composite": 1.0},
            "threshold": 4.0,
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_store("ok thanks")
    assert "Rejected" in result
    assert "quality gate" in result
    assert "4.0" in result


def test_store_error(mcp_server):
    api_resp = _make_response({"ok": False, "error": "Embedding failed"})
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_store("anything")
    assert "Error" in result
    assert "Embedding failed" in result


# -- memory_search ----------------------------------------------------------


def test_search_with_results(mcp_server):
    api_resp = _make_response(
        {
            "query": "database",
            "elapsed_ms": 55.2,
            "results": [
                {
                    "text": "We chose PostgreSQL for auth",
                    "final_score": 0.92,
                    "date": "2026-04-12T00:00:00Z",
                    "source": "decision",
                    "point_id": 123,
                },
                {
                    "text": "MySQL was considered but rejected",
                    "rerank_score": 0.71,
                    "date": "2026-04-11T00:00:00Z",
                    "source": "conversation",
                    "point_id": 456,
                },
            ],
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_search("database decisions", limit=5)
    assert "Found 2 memories" in result
    assert "55.2" in result
    assert "PostgreSQL" in result
    assert "score=0.920" in result
    assert "id=123" in result


def test_search_empty(mcp_server):
    api_resp = _make_response({"query": "nothing", "elapsed_ms": 10, "results": []})
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_search("xyzzy nonexistent")
    assert result == "No matching memories found."


def test_search_limit_clamped(mcp_server):
    captured_url = {}

    def fake_urlopen(req, **kwargs):
        captured_url["url"] = req.full_url
        return _make_response({"results": [], "elapsed_ms": 0})

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server.memory_search("q", limit=999)
    assert "limit=30" in captured_url["url"]

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server.memory_search("q", limit=-5)
    assert "limit=1" in captured_url["url"]


def test_search_score_fallback_chain(mcp_server):
    api_resp = _make_response(
        {
            "elapsed_ms": 10,
            "results": [{"text": "hit", "score": 0.5, "date": "", "source": "", "point_id": 1}],
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_search("q")
    assert "score=0.500" in result


# -- memory_reflect ---------------------------------------------------------


def test_reflect_formats_answer(mcp_server):
    api_resp = _make_response(
        {
            "answer": "PostgreSQL was chosen for the auth service.",
            "sources": [
                {"point_id": 10, "text": "We chose PostgreSQL", "score": 0.9},
                {"point_id": 11, "text": "MySQL rejected", "score": 0.7},
            ],
            "search_elapsed_ms": 45,
            "reflect_model": "claude-haiku",
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_reflect("What database did we pick?", limit=10)
    assert "PostgreSQL was chosen" in result
    assert "Sources (2 memories" in result
    assert "claude-haiku" in result
    assert "45ms" in result


# -- memory_stats -----------------------------------------------------------


def test_stats(mcp_server):
    api_resp = _make_response(
        {
            "qdrant": {"collection": "second_brain", "points": 50000},
            "graph": {"nodes": 1200, "edges": 3400},
            "status": "ok",
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_stats()
    assert "50000 memories" in result
    assert "second_brain" in result
    assert "1200 entity nodes" in result
    assert "3400 edges" in result
    assert "ok" in result


# -- memory_feedback --------------------------------------------------------


def test_feedback_helpful(mcp_server):
    api_resp = _make_response(
        {
            "ok": True,
            "point_id": 789,
            "helpful": True,
            "importance_before": 60,
            "importance_after": 65,
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_feedback("789", helpful=True)
    assert "helpful" in result
    assert "60 -> 65" in result


def test_feedback_not_helpful(mcp_server):
    api_resp = _make_response(
        {
            "ok": True,
            "point_id": 789,
            "helpful": False,
            "importance_before": 60,
            "importance_after": 50,
        }
    )
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_feedback("789", helpful=False)
    assert "not helpful" in result


def test_feedback_not_found(mcp_server):
    api_resp = _make_response({"ok": False, "error": "point_not_found"})
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_feedback("999", helpful=True)
    assert "Error" in result
    assert "point_not_found" in result


def test_feedback_converts_point_id_to_int(mcp_server):
    captured_body = {}

    def fake_urlopen(req, **kwargs):
        captured_body["data"] = json.loads(req.data.decode())
        return _make_response({"ok": True, "importance_before": 50, "importance_after": 55})

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server.memory_feedback("12345", helpful=True)
    assert captured_body["data"]["point_id"] == 12345
    assert isinstance(captured_body["data"]["point_id"], int)


def test_feedback_keeps_non_numeric_id_as_str(mcp_server):
    captured_body = {}

    def fake_urlopen(req, **kwargs):
        captured_body["data"] = json.loads(req.data.decode())
        return _make_response({"ok": True, "importance_before": 50, "importance_after": 55})

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server.memory_feedback("abc-uuid", helpful=True)
    assert captured_body["data"]["point_id"] == "abc-uuid"


# -- memory_commit_conversation ---------------------------------------------


def test_commit_conversation(mcp_server):
    api_resp = _make_response(
        {
            "ok": True,
            "turns_committed": 3,
            "windows_committed": 1,
            "total": 4,
        }
    )
    turns = [
        {"speaker": "Alice", "text": "Should we use Postgres?"},
        {"speaker": "Bob", "text": "Yes, better JSON support."},
        {"speaker": "Alice", "text": "Agreed, let's go with it."},
    ]
    with patch.object(mcp_server.urllib.request, "urlopen", return_value=api_resp):
        result = mcp_server.memory_commit_conversation(turns, source="meeting")
    assert "3 turns" in result
    assert "1 windows" in result
    assert "total: 4" in result


def test_commit_conversation_passes_params(mcp_server):
    captured_body = {}

    def fake_urlopen(req, **kwargs):
        captured_body["data"] = json.loads(req.data.decode())
        return _make_response({"ok": True, "turns_committed": 0, "windows_committed": 0, "total": 0})

    turns = [{"speaker": "A", "text": "hi"}]
    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server.memory_commit_conversation(turns, source="chat", window_size=3, stride=1)
    assert captured_body["data"]["source"] == "chat"
    assert captured_body["data"]["window_size"] == 3
    assert captured_body["data"]["stride"] == 1


# -- _api helper ------------------------------------------------------------


def test_api_sends_auth_header(mcp_server, monkeypatch):
    monkeypatch.setattr(mcp_server, "RASPUTIN_TOKEN", "secret-token")
    captured_headers = {}

    def fake_urlopen(req, **kwargs):
        captured_headers["auth"] = req.get_header("Authorization")
        return _make_response({"status": "ok"})

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server._api("/stats")
    assert captured_headers["auth"] == "Bearer secret-token"


def test_api_no_auth_header_when_empty(mcp_server, monkeypatch):
    monkeypatch.setattr(mcp_server, "RASPUTIN_TOKEN", "")
    captured_headers = {}

    def fake_urlopen(req, **kwargs):
        captured_headers["auth"] = req.get_header("Authorization")
        return _make_response({"status": "ok"})

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server._api("/stats")
    assert captured_headers["auth"] is None


# -- _collection_params -----------------------------------------------------


def test_collection_params_empty_by_default(mcp_server, monkeypatch):
    monkeypatch.setattr(mcp_server, "BANK_ID", "")
    assert mcp_server._collection_params() == {}


def test_collection_params_default_ignored(mcp_server, monkeypatch):
    monkeypatch.setattr(mcp_server, "BANK_ID", "default")
    assert mcp_server._collection_params() == {}


def test_collection_params_custom_bank(mcp_server, monkeypatch):
    monkeypatch.setattr(mcp_server, "BANK_ID", "my-project")
    assert mcp_server._collection_params() == {"collection": "my-project"}


def test_search_includes_collection_param(mcp_server, monkeypatch):
    monkeypatch.setattr(mcp_server, "BANK_ID", "custom-bank")
    captured_url = {}

    def fake_urlopen(req, **kwargs):
        captured_url["url"] = req.full_url
        return _make_response({"results": [], "elapsed_ms": 0})

    with patch.object(mcp_server.urllib.request, "urlopen", side_effect=fake_urlopen):
        mcp_server.memory_search("test")
    assert "collection=custom-bank" in captured_url["url"]
