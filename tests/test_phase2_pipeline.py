import json
from pathlib import Path

from tools.pipeline import query_expansion
from tools.pipeline.source_tiering import get_source_weight


def test_query_expansion_basic():
    queries = query_expansion.expand_queries(
        "What did we email about crypto last week?",
        max_expansions=10,
    )

    assert queries
    assert queries[0] == "What did we email about crypto last week?"
    assert any(q.startswith("email ") for q in queries)
    assert any(q.startswith("recent ") for q in queries)
    assert any("Bitcoin CHRONOS wallet blockchain hardware" in q for q in queries)


def test_query_expansion_entity_aware(tmp_path):
    graph_path = Path(tmp_path) / "entity_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "people": {
                    "Oren": {
                        "role": "founder",
                        "context": "chronos wallet",
                    }
                }
            }
        )
    )

    query_expansion.ENTITY_GRAPH_PATH = str(graph_path)
    queries = query_expansion.expand_queries("What did Oren discuss?", max_expansions=10)

    assert any("Oren founder chronos wallet" in q for q in queries)


def test_source_tiering_gold():
    assert get_source_weight("conversation") >= 0.9
    assert get_source_weight("email") >= 0.9


def test_source_tiering_unknown_source():
    assert get_source_weight("not_a_real_source") == 0.5
