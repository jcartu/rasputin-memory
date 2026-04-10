import importlib
import io
import json
from types import SimpleNamespace


state = importlib.import_module("brain._state")
embedding = importlib.import_module("brain.embedding")
entities = importlib.import_module("brain.entities")
commit_module = importlib.import_module("brain.commit")
amac = importlib.import_module("brain.amac")
server = importlib.import_module("brain.server")


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self._text = text
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def test_amac_parser_extracts_triplet(monkeypatch):
    monkeypatch.setattr(state.requests, "post", lambda *a, **k: FakeResponse("SCORES: 8,7,6"))
    scores = amac.amac_score("test memory")
    assert scores == (8.0, 7.0, 6.0, 7.0)


def test_amac_parser_ignores_examples(monkeypatch):
    raw = "example 0,1,0\nexample 4,2,2\nfinal answer\n9,8,7"
    monkeypatch.setattr(state.requests, "post", lambda *a, **k: FakeResponse(raw))
    scores = amac.amac_score("test memory")
    assert scores == (9.0, 8.0, 7.0, 8.0)


def test_dedup_detects_near_duplicate(monkeypatch):
    p = SimpleNamespace(id=101, score=0.96, payload={"text": "John Doe moved to Toronto yesterday"})
    monkeypatch.setattr(state, "qdrant", SimpleNamespace(query_points=lambda **k: SimpleNamespace(points=[p])))
    is_dupe, existing_id, similarity = embedding.check_duplicate([0.1] * 768, "John Doe moved to Toronto yesterday")
    assert is_dupe is True
    assert existing_id == 101
    assert similarity >= 0.95


def test_dedup_allows_different_text(monkeypatch):
    p = SimpleNamespace(id=102, score=0.93, payload={"text": "Completely unrelated business update"})
    monkeypatch.setattr(state, "qdrant", SimpleNamespace(query_points=lambda **k: SimpleNamespace(points=[p])))
    is_dupe, existing_id, similarity = embedding.check_duplicate([0.1] * 768, "Health appointment details with doctor")
    assert is_dupe is False
    assert existing_id is None
    assert similarity == 0


def test_payload_has_required_fields(monkeypatch, mock_qdrant):
    monkeypatch.setattr(state, "qdrant", mock_qdrant)
    monkeypatch.setattr(embedding, "get_embedding", lambda *a, **k: [0.2] * 768)
    monkeypatch.setattr(embedding, "check_duplicate", lambda *a, **k: (False, None, 0))
    monkeypatch.setattr(commit_module, "check_contradictions", lambda *a, **k: [])
    monkeypatch.setattr(entities, "extract_entities_fast", lambda *_: [])

    result = commit_module.commit_memory("A" * 64, source="conversation", importance=77)
    assert result["ok"] is True

    pid = result["id"]
    payload = mock_qdrant.points[pid]["payload"]
    for key in ("text", "source", "date", "importance", "embedding_model", "schema_version"):
        assert key in payload


def test_importance_clamped(monkeypatch):
    captured = {}

    def fake_commit_memory(text, source="conversation", importance=60, metadata=None):
        captured["importance"] = importance
        return {"ok": True, "id": 1}

    monkeypatch.setattr(commit_module, "commit_memory", fake_commit_memory)
    monkeypatch.setattr(amac, "amac_gate", lambda *a, **k: (True, "bypassed", {}))

    body = json.dumps({"text": "x" * 50, "importance": 999}).encode()
    handler = SimpleNamespace(
        path="/commit",
        headers={"Content-Length": str(len(body))},
        rfile=io.BytesIO(body),
        _check_auth=lambda: True,
        _enforce_rate_limit=lambda _endpoint: True,
    )
    sent = {}
    handler._send_json = lambda payload, status=200: sent.update({"payload": payload, "status": status})

    server.HybridHandler._handle_post(handler)
    assert sent["status"] == 200
    assert captured["importance"] == 100


def test_protected_fields_not_overwritten(monkeypatch, mock_qdrant):
    monkeypatch.setattr(state, "qdrant", mock_qdrant)
    monkeypatch.setattr(embedding, "get_embedding", lambda *a, **k: [0.3] * 768)
    monkeypatch.setattr(embedding, "check_duplicate", lambda *a, **k: (False, None, 0))
    monkeypatch.setattr(commit_module, "check_contradictions", lambda *a, **k: [])
    monkeypatch.setattr(entities, "extract_entities_fast", lambda *_: [])

    result = commit_module.commit_memory(
        "B" * 80,
        source="conversation",
        importance=61,
        metadata={
            "text": "override",
            "source": "bad",
            "importance": 1,
            "schema_version": "0.1",
            "custom_tag": "ok",
        },
    )
    payload = mock_qdrant.points[result["id"]]["payload"]
    assert payload["text"] != "override"
    assert payload["source"] == "conversation"
    assert payload["importance"] == 61
    assert payload["schema_version"] == "0.8"
    assert payload["custom_tag"] == "ok"


def test_rate_limiter_blocks_excess(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(server.time, "time", lambda: now["t"])

    limiter = server.SimpleRateLimiter(calls_per_minute=2)
    assert limiter.allow("/search:127.0.0.1") is True
    assert limiter.allow("/search:127.0.0.1") is True
    assert limiter.allow("/search:127.0.0.1") is False

    now["t"] += 61
    assert limiter.allow("/search:127.0.0.1") is True
