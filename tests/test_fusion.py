"""Phase C RRF fusion tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from brain.fusion import reciprocal_rank_fusion


def _row(pid, text="", score=0.5):
    return {"point_id": pid, "text": text, "score": score}


def test_rrf_rewards_cross_lane_agreement():
    """A doc appearing in 2 lanes at rank 1 beats a doc appearing in 1 lane at rank 1."""
    result_lists = {
        "lane_a": [_row("doc_shared", "shared"), _row("doc_a_only", "a_only")],
        "lane_b": [_row("doc_shared", "shared"), _row("doc_b_only", "b_only")],
    }
    merged = reciprocal_rank_fusion(result_lists, k=60)
    # doc_shared: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
    # doc_a_only: 1/(60+2) = 1/62 ≈ 0.0161
    # doc_b_only: 1/(60+2) = 1/62 ≈ 0.0161
    assert merged[0]["point_id"] == "doc_shared"
    assert merged[0]["rrf_rank"] == 1
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]
    assert set(merged[0]["source_ranks"].keys()) == {"lane_a", "lane_b"}


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion({}, k=60) == []
    assert reciprocal_rank_fusion({"a": [], "b": []}, k=60) == []


def test_rrf_dedupes_by_point_id():
    """Same point_id across lanes should be fused into one entry, not duplicated."""
    result_lists = {
        "lane_a": [_row("pid_1", "text_from_a")],
        "lane_b": [_row("pid_1", "text_from_b")],
    }
    merged = reciprocal_rank_fusion(result_lists, k=60)
    assert len(merged) == 1
    # First-seen instance wins the payload (lane_a registered it first)
    assert merged[0]["text"] == "text_from_a"


def test_rrf_k_constant_controls_tie_spread():
    """Lower k amplifies rank-1 advantage; higher k flattens."""
    rlists = {
        "a": [_row("p1"), _row("p2")],
    }
    # k=1: rank-1 worth 0.5, rank-2 worth 0.333 → 0.167 gap
    # k=60: rank-1 worth 1/61, rank-2 worth 1/62 → 0.00026 gap
    m_low = reciprocal_rank_fusion(rlists, k=1)
    m_hi = reciprocal_rank_fusion(rlists, k=60)
    gap_low = m_low[0]["rrf_score"] - m_low[1]["rrf_score"]
    gap_hi = m_hi[0]["rrf_score"] - m_hi[1]["rrf_score"]
    assert gap_low > gap_hi * 100  # low-k gap is at least 100x larger
