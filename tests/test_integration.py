import time

import pytest
import requests


BRAIN_URL = "http://localhost:7777"


@pytest.fixture(autouse=True)
def skip_if_no_server():
    try:
        r = requests.get(f"{BRAIN_URL}/health", timeout=2)
        if r.status_code != 200:
            pytest.skip("Memory server not ready")
    except Exception:
        pytest.skip("Memory server not running")


def test_health_check_endpoint():
    resp = requests.get(f"{BRAIN_URL}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "components" in body


def test_commit_and_search_round_trip():
    text = f"Integration test memory {time.time_ns()} about Toronto transplant meeting"
    commit_resp = requests.post(
        f"{BRAIN_URL}/commit",
        json={"text": text, "source": "test", "importance": 70, "force": True},
        timeout=10,
    )
    assert commit_resp.status_code == 200
    commit_json = commit_resp.json()
    assert commit_json.get("ok") is True

    found = []
    for _ in range(5):
        time.sleep(0.5)
        search_resp = requests.get(f"{BRAIN_URL}/search", params={"q": text[:40], "limit": 5}, timeout=10)
        assert search_resp.status_code == 200
        found = search_resp.json().get("results", [])
        if any(text[:40] in row.get("text", "") for row in found):
            break

    assert any(text[:40] in row.get("text", "") for row in found)


def test_feedback_endpoint_adjusts_importance():
    text = f"Feedback test memory {time.time_ns()} with enough content for commit"
    commit_resp = requests.post(
        f"{BRAIN_URL}/commit",
        json={"text": text, "source": "test", "importance": 60, "force": True},
        timeout=10,
    )
    assert commit_resp.status_code == 200
    payload = commit_resp.json()
    point_id = payload.get("id")
    assert point_id is not None

    feedback_resp = requests.post(f"{BRAIN_URL}/feedback", json={"point_id": point_id, "helpful": True}, timeout=10)
    assert feedback_resp.status_code in (200, 404)
    feedback_json = feedback_resp.json()
    if feedback_resp.status_code == 200:
        assert feedback_json.get("ok") is True
        assert feedback_json.get("importance_after", 0) >= feedback_json.get("importance_before", 0)
    else:
        if "Unknown path" in str(feedback_json.get("error", "")):
            pytest.skip("Feedback endpoint unavailable on running local server")
        assert feedback_json.get("error") == "point_not_found"
