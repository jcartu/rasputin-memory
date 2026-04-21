from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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


def _legacy_fact_payload() -> dict[str, object]:
    return {
        "what": "Alice joined Acme Corp on March 3, 2026",
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


def test_causal_link_pydantic_validation() -> None:
    link = fe.CausalLink(target_fact_id="F2", link_type="causes", weight=0.9)

    assert link.target_fact_id == "F2"
    assert link.link_type == "causes"
    assert link.weight == 0.9

    with pytest.raises(ValidationError):
        fe.CausalLink(target_fact_id="F2", link_type="invalid")

    with pytest.raises(ValidationError):
        fe.CausalLink(target_fact_id="F2", link_type="causes", weight=1.1)


def test_extracted_fact_defaults_backward_compat() -> None:
    fact = fe.ExtractedFact.model_validate(_legacy_fact_payload())

    assert fact.fact_id is None
    assert fact.causal_links == []
    assert fact.what == "Alice joined Acme Corp on March 3, 2026"


def test_extracted_fact_with_new_fields() -> None:
    fact = fe.ExtractedFact.model_validate(
        {
            "fact_id": "F1",
            "what": "Alice finished the migration",
            "who": "Alice",
            "when": "March 4, 2026",
            "where": "N/A",
            "fact_type": "world",
            "occurred_start": "2026-03-04",
            "occurred_end": None,
            "entities": [{"name": "Alice", "type": "Person"}],
            "causal_links": [{"target_fact_id": "F2", "link_type": "enables", "weight": 0.8}],
            "confidence": 0.9,
        }
    )

    dumped = fact.model_dump()

    assert dumped["fact_id"] == "F1"
    assert dumped["causal_links"] == [{"target_fact_id": "F2", "link_type": "enables", "weight": 0.8}]


def test_extract_causal_links_basic() -> None:
    facts = [
        {
            "fact_id": "F1",
            "what": "Deployment finished",
            "causal_links": [{"target_fact_id": "F2", "link_type": "causes", "weight": 0.9}],
        },
        {"fact_id": "F2", "what": "Customers regained access", "causal_links": []},
    ]

    assert fe.extract_causal_links_for_commit(facts) == [
        {"from_fact_id": "F1", "to_fact_id": "F2", "link_type": "causes", "weight": 0.9}
    ]


def test_extract_causal_links_missing_target() -> None:
    facts = [
        {
            "fact_id": "F1",
            "what": "Deployment finished",
            "causal_links": [{"target_fact_id": "F99", "link_type": "causes", "weight": 0.9}],
        }
    ]

    assert fe.extract_causal_links_for_commit(facts) == []


def test_extract_causal_links_self_link_dropped() -> None:
    facts = [
        {
            "fact_id": "F1",
            "what": "Deployment finished",
            "causal_links": [{"target_fact_id": "F1", "link_type": "causes", "weight": 0.9}],
        }
    ]

    assert fe.extract_causal_links_for_commit(facts) == []


def test_extract_causal_links_malformed() -> None:
    malformed_batches = [
        [{"fact_id": "F1", "what": "Deployment finished"}],
        [{"fact_id": "F1", "what": "Deployment finished", "causal_links": "not-a-list"}],
    ]

    for facts in malformed_batches:
        assert fe.extract_causal_links_for_commit(facts) == []


def test_verbose_prompt_contains_required_sections() -> None:
    prompt = fe.VERBOSE_EXTRACTION_PROMPT

    assert "5-10" in prompt
    assert "CAUSAL LINKS" in prompt
    assert "fact_id" in prompt
    assert "{event_date}" in prompt
    assert "{text}" in prompt

    for link_type in ("causes", "caused_by", "enables", "prevents"):
        assert link_type in prompt


def test_extract_facts_uses_verbose_prompt(tmp_path: Path) -> None:
    captured_prompts: list[str] = []

    def urlopen_stub(req, timeout: int = 0) -> MockHTTPResponse:
        del timeout
        body = json.loads(req.data.decode())
        captured_prompts.append(body["messages"][0]["content"])
        return MockHTTPResponse(_openai_payload([_legacy_fact_payload()]))

    with (
        patch.object(fe, "FACT_EXTRACTION_PROVIDER", "local_vllm"),
        patch.object(fe, "CEREBRAS_API_KEY", ""),
        patch.object(fe, "GROQ_API_KEY", ""),
        patch.object(fe, "ANTHROPIC_API_KEY", ""),
        patch.object(fe, "_EXTRACTION_PROVIDER_LOG", str(tmp_path / "extraction_provider_log.jsonl")),
        patch("brain.fact_extractor.request.urlopen", side_effect=urlopen_stub),
    ):
        facts = fe.extract_facts(
            "Alice joined Acme Corp on March 3, 2026 and said the move unlocked a new product launch.",
            event_date="2026-03-04",
            source="session-verbose",
        )

    assert len(captured_prompts) == 1
    assert any(marker in captured_prompts[0] for marker in ("5-10", "causal_links"))
    assert "Event Date: 2026-03-04" in captured_prompts[0]
    assert facts[0]["fact_id"] is None
    assert facts[0]["causal_links"] == []
