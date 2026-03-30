"""
Tests for the hybrid_brain library package.
No external services required — uses mocks throughout.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── RRFFusion ──────────────────────────────────────────────────────────────

class TestRRFFusion:
    def test_import(self):
        from hybrid_brain import RRFFusion
        assert RRFFusion is not None

    def test_basic_fusion(self):
        from hybrid_brain import RRFFusion
        fuser = RRFFusion(k=60)
        result = fuser.fuse([["a", "b", "c"], ["b", "a", "c"]])
        # a and b both appear in top ranks — should both be in result
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_consensus_wins(self):
        from hybrid_brain import RRFFusion
        fuser = RRFFusion(k=60)
        result = fuser.fuse([["a", "b"], ["a", "c"]])
        assert result[0] == "a"  # a is rank-1 in both lists

    def test_empty_lists(self):
        from hybrid_brain import RRFFusion
        fuser = RRFFusion()
        assert fuser.fuse([]) == []
        assert fuser.fuse([[]]) == []

    def test_fuse_with_scores(self):
        from hybrid_brain import RRFFusion
        fuser = RRFFusion(k=60)
        scored = fuser.fuse_with_scores([["x", "y"]])
        assert len(scored) == 2
        assert all(isinstance(s, float) for _, s in scored)
        assert scored[0][1] > scored[1][1]  # x ranked above y

    def test_k_effect(self):
        """Higher k should make score differences smaller."""
        from hybrid_brain import RRFFusion
        low_k = RRFFusion(k=1).fuse_with_scores([["a", "b"]])
        high_k = RRFFusion(k=1000).fuse_with_scores([["a", "b"]])
        diff_low = low_k[0][1] - low_k[1][1]
        diff_high = high_k[0][1] - high_k[1][1]
        assert diff_low > diff_high


# ── TemporalDecay ──────────────────────────────────────────────────────────

class TestTemporalDecay:
    def _make_result(self, score=0.9, days_ago=10, importance=50, source="conversation"):
        from datetime import datetime, timedelta
        return {
            "id": "test",
            "score": score,
            "text": "test memory",
            "source": source,
            "date": (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S"),
            "importance": importance,
            "retrieval_count": 0,
        }

    def test_import(self):
        from hybrid_brain import TemporalDecay
        assert TemporalDecay is not None

    def test_recent_high_importance_preserved(self):
        from hybrid_brain import TemporalDecay
        decay = TemporalDecay()
        r = self._make_result(score=0.9, days_ago=1, importance=90)
        results = decay.apply([r])
        # Score should be mostly preserved (recent, high importance)
        assert results[0]["score"] > 0.85

    def test_old_low_importance_decays(self):
        from hybrid_brain import TemporalDecay
        decay = TemporalDecay()
        r = self._make_result(score=0.9, days_ago=365, importance=20)
        results = decay.apply([r])
        # Should decay significantly
        assert results[0]["score"] < 0.5

    def test_floor_at_20_percent(self):
        from hybrid_brain import TemporalDecay
        decay = TemporalDecay()
        r = self._make_result(score=1.0, days_ago=10000, importance=1)
        results = decay.apply([r])
        # Never below 20% of original score
        assert results[0]["score"] >= 0.2 * 0.99  # small float tolerance

    def test_missing_date_no_crash(self):
        from hybrid_brain import TemporalDecay
        decay = TemporalDecay()
        r = {"id": "x", "score": 0.5, "text": "no date", "source": "test", "date": "", "importance": 50}
        results = decay.apply([r])
        assert len(results) == 1
        assert results[0]["score"] == 0.5  # unchanged

    def test_multifactor_adds_field(self):
        from hybrid_brain import TemporalDecay
        decay = TemporalDecay()
        r = self._make_result()
        results = decay.multifactor([r])
        assert "multifactor" in results[0]
        assert 0 < results[0]["multifactor"] <= 1.0

    def test_sort_order(self):
        from hybrid_brain import TemporalDecay
        decay = TemporalDecay()
        r1 = self._make_result(score=0.9, days_ago=1, importance=80)
        r2 = self._make_result(score=0.7, days_ago=500, importance=10)
        results = decay.apply([r2, r1])  # deliberately out of order
        assert results[0]["score"] >= results[1]["score"]


# ── QualityGate ────────────────────────────────────────────────────────────

class TestQualityGate:
    def test_import(self):
        from hybrid_brain import QualityGate
        assert QualityGate is not None

    def test_disabled_gate_admits_all(self):
        from hybrid_brain import QualityGate
        gate = QualityGate(enabled=False)
        result = gate.evaluate("gibberish xyz 123")
        assert result.admitted is True
        assert result.score == 10.0

    def test_gate_result_fields(self):
        from hybrid_brain import QualityGate
        gate = QualityGate(enabled=False)
        result = gate.evaluate("hello")
        assert hasattr(result, "admitted")
        assert hasattr(result, "score")
        assert hasattr(result, "reason")

    def test_llm_error_admits_by_default(self):
        from hybrid_brain import QualityGate
        gate = QualityGate(threshold=4.0)
        with patch("hybrid_brain.quality_gate.requests.post", side_effect=Exception("timeout")):
            result = gate.evaluate("some text")
        assert result.admitted is True  # fail-open
        assert "llm_error" in result.reason

    def test_threshold_respected(self):
        from hybrid_brain import QualityGate
        from unittest.mock import patch, MagicMock
        gate = QualityGate(threshold=5.0)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"score": 3.0, "reason": "low quality"}'}
        with patch("hybrid_brain.quality_gate.requests.post", return_value=mock_resp):
            result = gate.evaluate("bland content")
        assert result.admitted is False
        assert result.score == 3.0

    def test_above_threshold_admitted(self):
        from hybrid_brain import QualityGate
        gate = QualityGate(threshold=4.0)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"score": 8.5, "reason": "specific fact"}'}
        with patch("hybrid_brain.quality_gate.requests.post", return_value=mock_resp):
            result = gate.evaluate("Josh scheduled a meeting with the board on March 30.")
        assert result.admitted is True
        assert result.score == 8.5

    def test_force_bypasses_threshold(self):
        from hybrid_brain import QualityGate
        gate = QualityGate(threshold=9.0)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"score": 2.0, "reason": "test"}'}
        with patch("hybrid_brain.quality_gate.requests.post", return_value=mock_resp):
            result = gate.evaluate("text", force=True)
        assert result.admitted is True


# ── HybridSearch (unit — no real Qdrant) ──────────────────────────────────

class TestHybridSearchUnit:
    def _make_search(self):
        """HybridSearch with mocked Qdrant and embed."""
        from hybrid_brain import HybridSearch
        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value.points = []
        search = HybridSearch.__new__(HybridSearch)
        from hybrid_brain.fusion import RRFFusion
        from hybrid_brain.temporal import TemporalDecay
        from hybrid_brain.search import _BM25
        search.qdrant = mock_qdrant
        search.collection = "test_collection"
        search.embed_url = "http://localhost:11434/api/embed"
        search.embed_model = "nomic-embed-text-v2-moe"
        search.reranker_url = "http://localhost:8006/rerank"
        search.falkordb_host = "localhost"
        search.falkordb_port = 6380
        search.graph_name = "brain"
        search._rrf = RRFFusion(k=60)
        search._bm25 = _BM25()
        search._decay = TemporalDecay()
        search._use_multifactor = True
        return search, mock_qdrant

    def test_import(self):
        from hybrid_brain import HybridSearch
        assert HybridSearch is not None

    def test_empty_results_on_no_hits(self):
        search, _ = self._make_search()
        mock_embed_resp = MagicMock()
        mock_embed_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        with patch("hybrid_brain.search.requests.post", return_value=mock_embed_resp):
            results = search.query("test query", limit=5)
        assert results == []

    def test_rrf_integration(self):
        from hybrid_brain import RRFFusion
        fuser = RRFFusion()
        # Simulate two rankings and verify fusion result
        r1 = ["doc_a", "doc_b", "doc_c"]
        r2 = ["doc_b", "doc_a", "doc_d"]
        fused = fuser.fuse([r1, r2])
        # doc_a and doc_b both appear top-2 in both lists
        assert set(fused[:2]) == {"doc_a", "doc_b"}


# ── Package structure ──────────────────────────────────────────────────────

class TestPackageStructure:
    def test_top_level_imports(self):
        import hybrid_brain
        assert hasattr(hybrid_brain, "HybridSearch")
        assert hasattr(hybrid_brain, "RRFFusion")
        assert hasattr(hybrid_brain, "TemporalDecay")
        assert hasattr(hybrid_brain, "QualityGate")

    def test_version(self):
        import hybrid_brain
        assert hybrid_brain.__version__ == "0.3.0"

    def test_individual_module_imports(self):
        from hybrid_brain.fusion import RRFFusion
        from hybrid_brain.temporal import TemporalDecay
        from hybrid_brain.quality_gate import QualityGate
        from hybrid_brain.search import HybridSearch
        assert all([RRFFusion, TemporalDecay, QualityGate, HybridSearch])

    def test_all_exports(self):
        import hybrid_brain
        for name in hybrid_brain.__all__:
            assert hasattr(hybrid_brain, name), f"Missing export: {name}"
