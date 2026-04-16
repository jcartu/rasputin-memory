from __future__ import annotations

import importlib
import json
from unittest.mock import patch, MagicMock


knn_links = importlib.import_module("brain.knn_links")


def _mock_urlopen(response_data):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestComputeLinksForPoint:
    def test_returns_similar_ids_excluding_self(self):
        response = {
            "result": {
                "points": [
                    {"id": 100, "score": 0.9},
                    {"id": 200, "score": 0.8},
                    {"id": 42, "score": 0.7},
                ]
            }
        }
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = knn_links.compute_links_for_point("test_col", 42, [0.1] * 768, top_k=5, threshold=0.5)

        assert result == [100, 200]
        assert 42 not in result

    def test_respects_threshold(self):
        response = {
            "result": {
                "points": [
                    {"id": 100, "score": 0.9},
                    {"id": 200, "score": 0.4},
                ]
            }
        }
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = knn_links.compute_links_for_point("test_col", 1, [0.1] * 768, top_k=5, threshold=0.5)

        assert result == [100]


class TestStoreLinks:
    def test_sends_correct_payload_update(self):
        response = {"status": "ok"}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(response)) as mock_open:
            knn_links.store_links("test_col", 42, [100, 200, 300])

        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["points"] == [42]
        assert body["payload"]["similar_ids"] == [100, 200, 300]
        assert "points/payload" in req.full_url


class TestExpandSeeds:
    def test_fetches_linked_facts_and_formats(self):
        payload_response = {
            "result": [
                {"id": 10, "payload": {"similar_ids": [100, 200]}},
                {"id": 20, "payload": {"similar_ids": [200, 300]}},
            ]
        }
        facts_response = {
            "result": [
                {
                    "id": 100,
                    "payload": {"text": "Fact A", "source": "test", "date": "2025-01-01", "chunk_type": "fact"},
                },
                {
                    "id": 300,
                    "payload": {"text": "Fact C", "source": "test", "date": "2025-01-02", "chunk_type": "fact"},
                },
            ]
        }
        call_count = 0

        def side_effect(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen(payload_response)
            return _mock_urlopen(facts_response)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = knn_links.expand_seeds("test_col", [10, 20], exclude_ids={10, 20})

        assert len(result) == 2
        assert all(r["origin"] == "knn_expansion" for r in result)
        assert all(r["score"] == 0.5 for r in result)
        point_ids = {r["point_id"] for r in result}
        assert point_ids == {100, 300}

    def test_handles_empty_similar_ids(self):
        payload_response = {
            "result": [
                {"id": 10, "payload": {"similar_ids": []}},
                {"id": 20, "payload": {}},
            ]
        }
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload_response)):
            result = knn_links.expand_seeds("test_col", [10, 20], exclude_ids=set())

        assert result == []

    def test_respects_exclude_ids(self):
        payload_response = {
            "result": [
                {"id": 10, "payload": {"similar_ids": [100, 200, 300]}},
            ]
        }
        facts_response = {
            "result": [
                {"id": 300, "payload": {"text": "Fact C", "source": "test", "chunk_type": "fact"}},
            ]
        }
        call_count = 0

        def side_effect(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen(payload_response)
            return _mock_urlopen(facts_response)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = knn_links.expand_seeds("test_col", [10], exclude_ids={100, 200})

        point_ids = {r["point_id"] for r in result}
        assert 100 not in point_ids
        assert 200 not in point_ids

    def test_returns_empty_for_no_seeds(self):
        result = knn_links.expand_seeds("test_col", [], exclude_ids=set())
        assert result == []
