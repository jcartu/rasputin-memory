from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest

from brain import fact_extractor as fe


class MockHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self) -> MockHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def _fact(what: str = "Alice joined Acme Corp on March 3, 2026") -> dict[str, object]:
    return {
        "what": what,
        "who": "Alice",
        "when": "March 3, 2026",
        "where": "N/A",
        "fact_type": "world",
        "occurred_start": "2026-03-03",
        "occurred_end": None,
        "entities": [
            {"name": "Alice", "type": "Person"},
            {"name": "Acme Corp", "type": "Organization"},
        ],
        "confidence": 0.9,
    }


def _openai_payload(facts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": json.dumps({"facts": facts})}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 21},
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def test_determinism_mocked_local_vllm(tmp_path):
    responses = [MockHTTPResponse(_openai_payload([_fact()])) for _ in range(3)]

    with (
        patch.object(fe, "FACT_EXTRACTION_PROVIDER", "local_vllm"),
        patch.object(fe, "CEREBRAS_API_KEY", ""),
        patch.object(fe, "GROQ_API_KEY", ""),
        patch.object(fe, "ANTHROPIC_API_KEY", ""),
        patch.object(fe, "_EXTRACTION_PROVIDER_LOG", str(tmp_path / "extraction_provider_log.jsonl")),
        patch("brain.fact_extractor.time.sleep", return_value=None),
        patch("brain.fact_extractor.request.urlopen", side_effect=responses),
    ):
        results = [
            fe.extract_facts(
                "Alice joined Acme Corp on March 3, 2026 and said the move was a major career step forward.",
                event_date="2026-03-04",
                source="session-determinism",
            )
            for _ in range(3)
        ]

    first = {fact["what"] for fact in results[0]}
    second = {fact["what"] for fact in results[1]}
    third = {fact["what"] for fact in results[2]}

    assert _jaccard(first, second) == 1.0
    assert _jaccard(first, third) == 1.0


def test_local_vllm_5xx_falls_through_to_cerebras(tmp_path):
    urls_called: list[str] = []

    def urlopen_stub(req, timeout=0):
        del timeout
        urls_called.append(req.full_url)
        if req.full_url == fe.LOCAL_VLLM_URL:
            raise HTTPError(req.full_url, 503, "service unavailable", hdrs=None, fp=None)
        if req.full_url == fe.CEREBRAS_URL:
            return MockHTTPResponse(_openai_payload([_fact("Alice received a promotion")]))
        raise AssertionError(f"unexpected URL: {req.full_url}")

    with (
        patch.object(fe, "FACT_EXTRACTION_PROVIDER", "local_vllm"),
        patch.object(fe, "CEREBRAS_API_KEY", "cerebras-test-key"),
        patch.object(fe, "GROQ_API_KEY", ""),
        patch.object(fe, "ANTHROPIC_API_KEY", ""),
        patch.object(fe, "_cerebras_consecutive_failures", 0),
        patch.object(fe, "_EXTRACTION_PROVIDER_LOG", str(tmp_path / "extraction_provider_log.jsonl")),
        patch("brain.fact_extractor.time.sleep", return_value=None),
        patch("brain.fact_extractor.request.urlopen", side_effect=urlopen_stub),
    ):
        facts = fe.extract_facts(
            "Alice received a promotion after leading the launch review in March 2026.",
            event_date="2026-03-05",
            source="session-fallback",
        )

    assert urls_called.count(fe.LOCAL_VLLM_URL) == 2
    assert fe.CEREBRAS_URL in urls_called
    assert facts[0]["what"] == "Alice received a promotion"


def test_all_providers_fail_raises(tmp_path):
    def urlopen_stub(req, timeout=0):
        del req, timeout
        raise URLError(ConnectionRefusedError("offline"))

    with (
        patch.object(fe, "FACT_EXTRACTION_PROVIDER", "local_vllm"),
        patch.object(fe, "CEREBRAS_API_KEY", "cerebras-test-key"),
        patch.object(fe, "GROQ_API_KEY", "groq-test-key"),
        patch.object(fe, "ANTHROPIC_API_KEY", "anthropic-test-key"),
        patch.object(fe, "_cerebras_consecutive_failures", 0),
        patch.object(fe, "_EXTRACTION_PROVIDER_LOG", str(tmp_path / "extraction_provider_log.jsonl")),
        patch("brain.fact_extractor.time.sleep", return_value=None),
        patch("brain.fact_extractor.request.urlopen", side_effect=urlopen_stub),
    ):
        with pytest.raises(RuntimeError, match="all extractors failed"):
            fe.extract_facts(
                "Alice discussed the launch timeline, vendor outage, and rollback plan in detail yesterday.",
                event_date="2026-03-06",
                source="session-failure",
            )
