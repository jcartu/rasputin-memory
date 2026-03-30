import importlib


bm25_search = importlib.import_module("bm25_search")
BM25Scorer = bm25_search.BM25Scorer
hybrid_rerank = bm25_search.hybrid_rerank
reciprocal_rank_fusion = bm25_search.reciprocal_rank_fusion


def test_tokenize_basic():
    tokens = BM25Scorer().tokenize("Hello, World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_cyrillic():
    tokens = BM25Scorer().tokenize("Москва — столица России")
    assert "москва" in tokens
    assert "россии" in tokens


def test_bm25_scoring_relevance():
    results = [
        {
            "score": 0.2,
            "payload": {"text": "The quick brown fox jumps over the lazy dog"},
        },
        {
            "score": 0.9,
            "payload": {"text": "Python programming language"},
        },
        {
            "score": 0.3,
            "payload": {"text": "Quick fox is fast"},
        },
    ]
    reranked = hybrid_rerank("quick fox", results, bm25_weight=1.0)
    top_text = reranked[0]["payload"]["text"].lower()
    assert "quick" in top_text
    assert "fox" in top_text


def test_rrf_equal_length():
    dense = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
    bm25_scores = [0.5, 0.7]
    result = reciprocal_rank_fusion(dense, bm25_scores)
    assert len(result) == 2
    assert all("rrf_score" in row for row in result)
    assert all("bm25_score" in row for row in result)


def test_rrf_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_hybrid_rerank_empty_inputs():
    assert hybrid_rerank("anything", []) == []
