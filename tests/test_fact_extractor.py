from __future__ import annotations

import importlib
import json
from datetime import datetime


def test_parse_llm_response_extracts_json_array():
    fact_extractor = importlib.import_module("fact_extractor")

    raw = """Here are facts:
```json
[{"category": "Business", "fact": "Revenue reached $2.3M in Q1 2026"}]
```
"""
    parsed = fact_extractor.parse_llm_response(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["category"] == "Business"


def test_purge_garbage_facts_filters_vague_entries(tmp_path, monkeypatch):
    fact_extractor = importlib.import_module("fact_extractor")

    facts_path = tmp_path / "facts.jsonl"
    facts_path.write_text(
        "\n".join(
            [
                json.dumps({"fact": "user likes music"}),
                json.dumps({"fact": "Alice joined Acme Corp on 2026-03-01"}),
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(fact_extractor, "FACTS_FILE", facts_path)
    purged = fact_extractor.purge_garbage_facts()

    remaining = [json.loads(line) for line in facts_path.read_text().splitlines() if line.strip()]
    assert purged == 1
    assert len(remaining) == 1
    assert "Alice joined Acme Corp" in remaining[0]["fact"]


def test_fact_dedup_hash_comparison_is_case_insensitive():
    fact_extractor = importlib.import_module("fact_extractor")

    is_dup_a, hash_a = fact_extractor.dedup_fact("Alice moved to London", set())
    is_dup_b, hash_b = fact_extractor.dedup_fact("  alice moved to london  ", {hash_a})

    assert is_dup_a is False
    assert is_dup_b is True
    assert hash_a == hash_b


def test_locking_uses_shared_pipeline_locking_module():
    fact_extractor = importlib.import_module("fact_extractor")

    assert fact_extractor.acquire_pipeline_lock.__name__ == "acquire_lock"
    assert fact_extractor.acquire_pipeline_lock.__module__.endswith("pipeline.locking")


def test_extract_chunk_pass_pipeline_and_store(tmp_path, monkeypatch):
    fact_extractor = importlib.import_module("fact_extractor")

    session_file = tmp_path / "session.jsonl"
    now = datetime.utcnow().isoformat()
    lines = [
        {
            "type": "message",
            "timestamp": now,
            "message": {
                "role": "user",
                "content": "Alice joined Acme Corp on March 3, 2026 and revenue reached $2.3M in Q1.",
            },
        },
        {
            "type": "message",
            "timestamp": now,
            "message": {
                "role": "assistant",
                "content": "Noted: hiring milestone and business performance update.",
            },
        },
    ]
    session_file.write_text("\n".join(json.dumps(row) for row in lines) + "\n")

    facts_file = tmp_path / "facts.jsonl"
    monkeypatch.setattr(fact_extractor, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(fact_extractor, "FACTS_FILE", facts_file)

    msgs = fact_extractor.extract_user_messages(hours=24, process_all=False)
    assert len(msgs) == 2

    chunks = fact_extractor.chunk_messages(msgs, chunk_size=2)
    assert len(chunks) == 1

    monkeypatch.setattr(
        fact_extractor,
        "llm_call",
        lambda *_args, **_kwargs: '[{"category":"Business","fact":"Alice joined Acme Corp on March 3, 2026"}]',
    )
    pass1 = fact_extractor.pass1_extract_facts(chunks[0]["text"])
    assert len(pass1) == 1

    monkeypatch.setattr(
        fact_extractor,
        "llm_call",
        lambda *_args, **_kwargs: (
            '[{"fact":"Alice joined Acme Corp on March 3, 2026","status":"CONFIRMED","reason":"present"}]'
        ),
    )
    pass2 = fact_extractor.pass2_verify_facts(pass1, chunks[0]["text"])
    assert len(pass2) == 1

    monkeypatch.setattr(
        fact_extractor,
        "llm_call",
        lambda *_args, **_kwargs: '["Alice joined Acme Corp on March 3, 2026"]',
    )
    pass3 = fact_extractor.pass3_filter_existing(pass2, existing_facts=[])
    assert len(pass3) == 1

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True, "amac": {"scores": {"composite": 8.1}}}

    monkeypatch.setattr(fact_extractor.requests, "post", lambda *_a, **_k: _Resp())
    state = {"fact_hashes": []}
    assert fact_extractor.store_fact(pass3[0], state) is True
    assert len(state["fact_hashes"]) == 1
    assert facts_file.exists()


def test_main_no_messages_saves_state_and_exits(monkeypatch, tmp_path):
    fact_extractor = importlib.import_module("fact_extractor")

    saved = {"called": False}
    monkeypatch.setattr(fact_extractor, "acquire_pipeline_lock", lambda _name: 1)
    monkeypatch.setattr(
        fact_extractor, "load_state", lambda: {"last_run": None, "processed_lines": {}, "fact_hashes": []}
    )
    monkeypatch.setattr(fact_extractor, "extract_user_messages", lambda **_k: [])
    monkeypatch.setattr(fact_extractor, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(fact_extractor, "FACTS_FILE", tmp_path / "facts.jsonl")

    def fake_save_state(state):
        saved["called"] = True
        assert state["last_run"] is not None

    monkeypatch.setattr(fact_extractor, "save_state", fake_save_state)
    monkeypatch.setattr(fact_extractor, "sys", type("_S", (), {"argv": ["fact_extractor.py"]})())

    fact_extractor.main()
    assert saved["called"] is True


def test_pydantic_model_valid():
    from brain.fact_extractor import ExtractedFact

    fact = ExtractedFact(
        what="Caroline attended an LGBTQ support group meeting",
        who="Caroline",
        when="May 7, 2023",
        where="Community Center",
        fact_type="world",
        occurred_start="2023-05-07",
        occurred_end=None,
        entities=[{"name": "Caroline", "type": "Person"}],
        confidence=0.9,
    )
    assert fact.what == "Caroline attended an LGBTQ support group meeting"
    assert fact.fact_type == "world"
    assert fact.confidence == 0.9
    assert len(fact.entities) == 1
    assert fact.entities[0].name == "Caroline"


def test_pydantic_model_defaults():
    from brain.fact_extractor import ExtractedFact

    fact = ExtractedFact(what="Some fact happened")
    assert fact.who == "N/A"
    assert fact.when == "N/A"
    assert fact.where == "N/A"
    assert fact.fact_type == "world"
    assert fact.occurred_start is None
    assert fact.occurred_end is None
    assert fact.entities == []
    assert fact.confidence == 0.8


def test_inference_extraction(monkeypatch):
    from brain import fact_extractor as fe
    import requests as _requests_mod

    mock_response = json.dumps(
        {
            "facts": [
                {
                    "what": "Caroline is likely supportive of LGBTQ causes",
                    "who": "Caroline",
                    "when": "N/A",
                    "where": "N/A",
                    "fact_type": "inference",
                    "occurred_start": None,
                    "occurred_end": None,
                    "entities": [{"name": "Caroline", "type": "Person"}],
                    "confidence": 0.7,
                },
            ]
        }
    )

    monkeypatch.setattr(fe, "FACT_EXTRACTION_PROVIDER", "test")
    monkeypatch.setattr(fe, "ANTHROPIC_API_KEY", "")

    class MockResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": mock_response}}]}

    monkeypatch.setattr(_requests_mod, "post", lambda *a, **k: MockResp())

    facts = fe.extract_facts(
        "Caroline mentioned she went to the LGBTQ support group.",
        event_date="2023-05-08",
    )
    inferences = [f for f in facts if f.get("fact_type") == "inference"]
    assert len(inferences) >= 1
    assert inferences[0]["confidence"] == 0.7


def test_temporal_range_parsing(monkeypatch):
    from brain import fact_extractor as fe
    import requests as _requests_mod

    mock_response = json.dumps(
        {
            "facts": [
                {
                    "what": "Alice worked at Google",
                    "who": "Alice",
                    "when": "2021 to 2024",
                    "where": "N/A",
                    "fact_type": "world",
                    "occurred_start": "2021-01-01",
                    "occurred_end": "2024-12-31",
                    "entities": [
                        {"name": "Alice", "type": "Person"},
                        {"name": "Google", "type": "Organization"},
                    ],
                    "confidence": 0.9,
                },
            ]
        }
    )

    monkeypatch.setattr(fe, "FACT_EXTRACTION_PROVIDER", "test")
    monkeypatch.setattr(fe, "ANTHROPIC_API_KEY", "")

    class MockResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": mock_response}}]}

    monkeypatch.setattr(_requests_mod, "post", lambda *a, **k: MockResp())

    facts = fe.extract_facts(
        "Alice worked at Google from 2021 to 2024.",
        event_date="2024-06-01",
    )
    assert len(facts) >= 1
    assert facts[0]["occurred_start"] == "2021-01-01"
    assert facts[0]["occurred_end"] == "2024-12-31"


def test_parse_response_valid_json():
    from brain.fact_extractor import _parse_extraction_response

    response = json.dumps(
        {
            "facts": [
                {
                    "what": "Test fact",
                    "who": "Alice",
                    "when": "2023",
                    "where": "N/A",
                    "fact_type": "world",
                    "occurred_start": "2023-01-01",
                    "occurred_end": None,
                    "entities": [],
                    "confidence": 0.9,
                }
            ]
        }
    )
    facts = _parse_extraction_response(response)
    assert len(facts) == 1
    assert facts[0]["what"] == "Test fact"
    assert facts[0]["fact_type"] == "world"


def test_parse_response_invalid_json():
    from brain.fact_extractor import _parse_extraction_response

    facts = _parse_extraction_response("This is not JSON at all")
    assert facts == []


def test_parse_response_old_format():
    from brain.fact_extractor import _parse_extraction_response

    response = '[{"what": "Old format fact", "who": "Bob", "when": "June 2024", "entities": ["Bob"]}]'
    facts = _parse_extraction_response(response)
    assert len(facts) == 1
    assert facts[0]["what"] == "Old format fact"


def test_date_range_extraction():
    from brain.search import _extract_date_range

    start, end = _extract_date_range("What happened in January 2023?")
    assert start == "2023-01-01"
    assert end == "2023-01-28"


def test_date_range_no_dates():
    from brain.search import _extract_date_range

    start, end = _extract_date_range("What is Alice's favorite color?")
    assert start is None
    assert end is None
