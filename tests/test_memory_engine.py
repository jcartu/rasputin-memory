from __future__ import annotations

import importlib
import requests


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_recall_makes_correct_get_request(monkeypatch):
    memory_engine = importlib.import_module("memory_engine")

    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse({"results": []})

    monkeypatch.setattr(memory_engine.requests, "get", fake_get)
    result = memory_engine.recall("release timeline", limit=7, expand=False)

    assert captured["url"].endswith("/search")
    assert captured["params"] == {"q": "release timeline", "limit": 7, "expand": "false"}
    assert captured["timeout"] == memory_engine.DEFAULT_TIMEOUT
    assert result == {"results": []}


def test_commit_makes_correct_post_request(monkeypatch):
    memory_engine = importlib.import_module("memory_engine")

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(memory_engine.requests, "post", fake_post)
    result = memory_engine.commit("Important memory", source="conversation", importance=77, metadata={"a": 1})

    assert captured["url"].endswith("/commit")
    assert captured["json"] == {
        "text": "Important memory",
        "source": "conversation",
        "importance": 77,
        "metadata": {"a": 1},
    }
    assert captured["timeout"] == memory_engine.DEFAULT_TIMEOUT
    assert result == {"ok": True}


def test_main_cli_dispatch_for_each_command(monkeypatch):
    memory_engine = importlib.import_module("memory_engine")

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(memory_engine, "recall", lambda text: calls.append(("recall", text)) or {"results": []})
    monkeypatch.setattr(memory_engine, "commit", lambda text: calls.append(("commit", text)) or {"ok": True})
    monkeypatch.setattr(memory_engine, "deep", lambda text: calls.append(("deep", text)) or {"results": []})
    monkeypatch.setattr(memory_engine, "whois", lambda text: calls.append(("whois", text)) or {"results": []})
    monkeypatch.setattr(memory_engine, "challenge", lambda text: calls.append(("challenge", text)) or {"results": []})
    monkeypatch.setattr(memory_engine, "briefing", lambda: calls.append(("briefing", "")) or {"results": []})
    monkeypatch.setattr(memory_engine, "_print_search_results", lambda _result: None)

    assert memory_engine.main(["recall", "alpha"]) == 0
    assert memory_engine.main(["commit", "beta"]) == 0
    assert memory_engine.main(["deep", "gamma"]) == 0
    assert memory_engine.main(["whois", "delta"]) == 0
    assert memory_engine.main(["challenge", "epsilon"]) == 0
    assert memory_engine.main(["briefing"]) == 0

    assert calls == [
        ("recall", "alpha"),
        ("commit", "beta"),
        ("deep", "gamma"),
        ("whois", "delta"),
        ("challenge", "epsilon"),
        ("briefing", ""),
    ]


def test_main_handles_request_failure(monkeypatch, capsys):
    memory_engine = importlib.import_module("memory_engine")

    def fake_get(*_args, **_kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(memory_engine.requests, "get", fake_get)
    rc = memory_engine.main(["recall", "failure case"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "Request failed:" in captured.err


def test_main_without_args_prints_usage_and_returns_1(capsys):
    memory_engine = importlib.import_module("memory_engine")

    rc = memory_engine.main([])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Usage: memory_engine.py" in captured.out
