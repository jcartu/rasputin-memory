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
