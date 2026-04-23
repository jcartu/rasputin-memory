"""Phase B four-partition retrieval tests (FOUR_LANE=1)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add tools/ to path so we can import brain.*
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


@pytest.fixture
def _enable_four_lane(monkeypatch):
    """Opt-in fixture: enable FOUR_LANE for tests that exercise the hybrid_search gate."""
    monkeypatch.setenv("FOUR_LANE", "1")
    yield


def _make_point(pid, text, chunk_type="fact", fact_type="world", score=0.8):
    p = MagicMock()
    p.id = pid
    p.score = score
    p.payload = {
        "text": text,
        "chunk_type": chunk_type,
        "fact_type": fact_type,
        "source": "test",
        "date": "2026-04-20",
    }
    return p


def test_four_lane_dispatches_four_parallel_lanes():
    """_four_lane_search must issue 4 parallel Qdrant lanes: window + world + exp + inf."""
    from brain import search as _search

    call_log: list[dict] = []

    def fake_qdrant_search(
        query,
        limit=10,
        source_filter=None,
        collection=None,
        chunk_type_filter=None,
        fact_type_filter=None,
    ):
        call_log.append(
            {
                "query": query,
                "limit": limit,
                "chunk_type_filter": chunk_type_filter,
                "fact_type_filter": fact_type_filter,
            }
        )
        return [
            {
                "point_id": f"{chunk_type_filter or 'none'}_{fact_type_filter or 'any'}",
                "text": f"doc for {chunk_type_filter}/{fact_type_filter}",
                "score": 0.5,
                "chunk_type": chunk_type_filter or "",
                "fact_type": fact_type_filter or "",
                "origin": "qdrant",
            }
        ]

    with patch.object(_search, "qdrant_search", side_effect=fake_qdrant_search):
        result = _search._four_lane_search("what is alice's job", collection=None, source_filter=None)

    # Four Qdrant lanes fired
    chunk_types = sorted(c.get("chunk_type_filter") for c in call_log)
    assert chunk_types == ["fact", "fact", "fact", "window"], f"Expected 4 Qdrant lanes, got {call_log}"

    fact_types = sorted(c.get("fact_type_filter") for c in call_log if c.get("chunk_type_filter") == "fact")
    assert fact_types == ["experience", "inference", "world"]

    # Lane dict returned with all 4 keys (no bm25 — see docs/BACKLOG.md)
    assert set(result.keys()) == {"window", "world", "exp", "inf"}


def test_four_lane_respects_budget_env_vars(monkeypatch):
    """BENCH_LANE_* env vars must be read at call time and passed as `limit` to each qdrant_search call."""
    monkeypatch.setenv("BENCH_LANE_WINDOWS", "99")
    monkeypatch.setenv("BENCH_LANE_FACT_W", "11")
    monkeypatch.setenv("BENCH_LANE_FACT_E", "7")
    monkeypatch.setenv("BENCH_LANE_FACT_I", "5")

    from brain import search as _search

    call_log: list[dict] = []

    def fake_qdrant_search(query, limit=10, **kw):
        call_log.append({"limit": limit, "ct": kw.get("chunk_type_filter"), "ft": kw.get("fact_type_filter")})
        return []

    with patch.object(_search, "qdrant_search", side_effect=fake_qdrant_search):
        _search._four_lane_search("q", collection=None, source_filter=None)

    limits_by_type = {(c["ct"], c["ft"]): c["limit"] for c in call_log}
    assert limits_by_type[("window", None)] == 99
    assert limits_by_type[("fact", "world")] == 11
    assert limits_by_type[("fact", "experience")] == 7
    assert limits_by_type[("fact", "inference")] == 5


def test_fact_type_filter_pushed_into_qdrant_filter():
    """qdrant_search(fact_type_filter='world') must add a FieldCondition on fact_type."""
    from brain import search as _search

    captured_filter = {}

    class FakePoints:
        points = []

    def fake_query_points(collection_name, query, limit, query_filter=None, with_payload=True):
        captured_filter["filter"] = query_filter
        return FakePoints()

    with patch.object(_search._state, "qdrant") as mock_q:
        mock_q.query_points.side_effect = fake_query_points
        with patch.object(_search.embedding, "get_embedding", return_value=[0.1] * 768):
            _search.qdrant_search("test", limit=5, chunk_type_filter="fact", fact_type_filter="world")

    f = captured_filter["filter"]
    assert f is not None, "Expected non-None Filter"
    # Two conditions: chunk_type=fact AND fact_type=world
    must_conds = f.must if hasattr(f, "must") else []
    keys = {c.key for c in must_conds if hasattr(c, "key")}
    assert "chunk_type" in keys
    assert "fact_type" in keys


def test_default_fourlane_off_does_not_invoke_four_lane_search(monkeypatch):
    """FOUR_LANE=0 must NOT call _four_lane_search — legacy codepath only."""
    monkeypatch.setenv("FOUR_LANE", "0")

    from brain import search as _search

    four_lane_called: list[bool] = []

    def mock_four_lane(*a, **kw):
        four_lane_called.append(True)
        return {}

    def fake_qdrant_search(query, **kw):
        return []

    def fake_graph(*a, **kw):
        return []

    with (
        patch.object(_search, "_four_lane_search", side_effect=mock_four_lane),
        patch.object(_search, "qdrant_search", side_effect=fake_qdrant_search),
        patch.object(_search.graph, "graph_search", side_effect=fake_graph),
        patch.object(_search.graph, "enrich_with_graph", return_value={}),
    ):
        _search.hybrid_search("q", limit=5, expand=False)

    assert four_lane_called == [], "FOUR_LANE=0 must NOT invoke _four_lane_search"


def test_fourlane_on_invokes_four_lane_search(_enable_four_lane):
    """FOUR_LANE=1 must route hybrid_search through _four_lane_search."""
    from brain import search as _search

    four_lane_called: list[bool] = []

    def mock_four_lane(*a, **kw):
        four_lane_called.append(True)
        return {"window": [], "world": [], "exp": [], "inf": []}

    def fake_qdrant_search(query, **kw):
        return []

    def fake_graph(*a, **kw):
        return []

    with (
        patch.object(_search, "_four_lane_search", side_effect=mock_four_lane),
        patch.object(_search, "qdrant_search", side_effect=fake_qdrant_search),
        patch.object(_search.graph, "graph_search", side_effect=fake_graph),
        patch.object(_search.graph, "enrich_with_graph", return_value={}),
    ):
        _search.hybrid_search("q", limit=5, expand=False)

    assert four_lane_called == [True], "FOUR_LANE=1 must invoke _four_lane_search exactly once"
