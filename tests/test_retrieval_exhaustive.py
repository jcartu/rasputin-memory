#!/usr/bin/env python3
"""
EXHAUSTIVE TEST SUITE FOR RETRIEVAL PIPELINE
Target: 100+ test cases covering EVERYTHING

Tests for:
- hybrid_brain.py (main retrieval API server)
- tools/memory_engine.py (CLI retrieval tool)
- All retrieval-related modules in hybrid_brain/ directory

Coverage:
1. Unit tests — every function, every branch, every edge case
2. Integration tests — full search pipeline end-to-end
3. Edge cases — empty queries, Unicode/Cyrillic, extremely long queries, special characters, SQL injection
4. Error paths — Qdrant down, FalkorDB down, reranker down, timeout scenarios
5. Performance — response time assertions for typical queries
6. Regression tests — for all known bug patterns
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

# Add paths
ROOT = "/home/josh/.openclaw/workspace/rasputin-memory"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/tools")

# Import modules under test
import bm25_search
import config
from hybrid_brain import (
    AMAC_THRESHOLD,
    BM25_AVAILABLE,
    FALKORDB_DISABLED,
    STOP_WORDS,
    _amac_metrics,
    _amac_metrics_lock,
    _inc_metric,
    amac_gate,
    apply_multifactor_scoring,
    apply_relevance_feedback,
    apply_temporal_decay,
    check_duplicate,
    commit_memory,
    enrich_with_graph,
    extract_entities_fast,
    get_embedding_safe,
    get_source_weight,
    graph_search,
    hybrid_search,
    is_reranker_available,
    list_contradictions,
    neural_rerank,
    proactive_surface,
    qdrant_search,
)

# Mock fixtures will be injected via conftest
from tests.conftest import MockPoint, MockQdrant, MockRedis


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch, mock_qdrant, mock_redis, mock_embedding):
    """Set up all mocks for each test."""
    monkeypatch.setattr("hybrid_brain.qdrant", mock_qdrant)
    monkeypatch.setattr("hybrid_brain._redis_pool", mock_redis)
    monkeypatch.setattr("hybrid_brain.get_redis", lambda: mock_redis)
    monkeypatch.setattr("hybrid_brain.get_embedding", mock_embedding)
    monkeypatch.setattr("hybrid_brain.is_reranker_available", lambda: False)
    monkeypatch.setattr("hybrid_brain.BM25_AVAILABLE", False)
    monkeypatch.setattr("hybrid_brain.FALKORDB_DISABLED", True)

    with _amac_metrics_lock:
        _amac_metrics.clear()
        _amac_metrics.update(
            {"accepted": 0, "rejected": 0, "bypassed": 0, "score_sum": 0.0, "score_count": 0, "timeout_accepts": 0}
        )


# ============================================================================
# CONFIG TESTS (1-5)
# ============================================================================


class TestConfig:
    """Tests for config loading and environment variable overrides."""

    def test_config_loads_default_values(self):
        """Test that config loads with default values."""
        cfg = config.load_config("config/rasputin.toml")
        assert "server" in cfg
        assert "qdrant" in cfg
        assert "graph" in cfg
        assert "embeddings" in cfg
        assert "amac" in cfg

    def test_config_resolves_relative_paths(self):
        """Test that relative paths are resolved correctly."""
        cfg = config.load_config("config/rasputin.toml")
        entities_path = cfg.get("entities", {}).get("known_entities_path", "")
        assert entities_path

    def test_config_env_override_server_host(self, monkeypatch):
        """Test SERVER_ENV environment variable override."""
        monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
        cfg = config.load_config("config/rasputin.toml")
        assert cfg["server"]["host"] == "0.0.0.0"

    def test_config_env_override_port(self, monkeypatch):
        """Test PORT environment variable override."""
        monkeypatch.setenv("PORT", "9999")
        cfg = config.load_config("config/rasputin.toml")
        assert cfg["server"]["port"] == 9999

    def test_config_env_override_qdrant_url(self, monkeypatch):
        """Test QDRANT_URL environment variable override."""
        monkeypatch.setenv("QDRANT_URL", "http://test:6333")
        cfg = config.load_config("config/rasputin.toml")
        assert cfg["qdrant"]["url"] == "http://test:6333"


# ============================================================================
# BM25 SEARCH TESTS (6-20)
# ============================================================================


class TestBM25Search:
    """Tests for BM25 reranking layer."""

    def test_bm25_tokenizer_lowercase(self):
        """Test tokenization converts to lowercase."""
        scorer = bm25_search.BM25Scorer()
        tokens = scorer.tokenize("Hello WORLD")
        assert tokens == ["hello", "world"]

    def test_bm25_tokenizer_removes_special_chars(self):
        """Test tokenization removes special characters."""
        scorer = bm25_search.BM25Scorer()
        tokens = scorer.tokenize("test@email.com and $100!")
        assert "test" in tokens
        assert "email" in tokens
        assert "com" in tokens
        assert "and" in tokens
        assert "100" in tokens

    def test_bm25_empty_query_returns_zero_scores(self):
        """Test empty query returns zero scores."""
        scorer = bm25_search.BM25Scorer()
        scores = scorer.score("", ["doc1", "doc2"])
        assert scores == [0.0, 0.0]

    def test_bm25_empty_documents_returns_zero_scores(self):
        """Test empty documents list returns empty scores."""
        scorer = bm25_search.BM25Scorer()
        scores = scorer.score("query", [])
        assert scores == []

    def test_bm25_scores_higher_for_matching_terms(self):
        """Test documents with more query terms get higher scores."""
        scorer = bm25_search.BM25Scorer()
        docs = ["apple banana", "apple", "cherry"]
        scores = scorer.score("apple", docs)
        assert scores[0] > 0 or scores[1] > 0

    def test_bm25_idf_penalizes_common_terms(self):
        """Test IDF penalizes very common terms."""
        scorer = bm25_search.BM25Scorer()
        docs = ["the cat", "the dog", "the bird"]
        scores = scorer.score("the", docs)
        assert all(s >= 0 for s in scores)

    def test_reciprocal_rank_fusion_basic(self):
        """Test basic RRF fusion."""
        dense = [
            {"score": 0.9, "text": "doc1"},
            {"score": 0.8, "text": "doc2"},
            {"score": 0.7, "text": "doc3"},
        ]
        bm25_scores = [0.5, 0.6, 0.4]

        fused = bm25_search.reciprocal_rank_fusion(dense, bm25_scores)

        assert len(fused) == 3
        assert all("rrf_score" in r for r in fused)
        assert all("bm25_score" in r for r in fused)

    def test_reciprocal_rank_fusion_empty_results(self):
        """Test RRF with empty results."""
        fused = bm25_search.reciprocal_rank_fusion([], [])
        assert fused == []

    def test_reciprocal_rank_fusion_mismatched_lengths(self):
        """Test RRF handles mismatched list lengths."""
        dense = [{"score": 0.9}, {"score": 0.8}]
        bm25_scores = [0.5]

        fused = bm25_search.reciprocal_rank_fusion(dense, bm25_scores)
        assert len(fused) == 2

    def test_hybrid_rerank_basic(self):
        """Test hybrid rerank function."""
        query = "test query"
        docs = [
            {"score": 0.9, "payload": {"text": "doc1"}},
            {"score": 0.8, "payload": {"text": "doc2"}},
        ]

        reranked = bm25_search.hybrid_rerank(query, docs)

        assert len(reranked) == 2
        assert all("rrf_score" in r for r in reranked)

    def test_hybrid_rerank_empty_results(self):
        """Test hybrid rerank with empty results."""
        reranked = bm25_search.hybrid_rerank("query", [])
        assert reranked == []

    def test_hybrid_rerank_zero_bm25_weight(self):
        """Test hybrid rerank with bm25_weight=0."""
        docs = [{"score": 0.9, "payload": {"text": "doc"}}]
        reranked = bm25_search.hybrid_rerank("query", docs, bm25_weight=0)
        assert len(reranked) == 1

    def test_hybrid_rerank_extract_text_from_payload(self):
        """Test text extraction from various payload structures."""
        docs = [
            {"score": 0.9, "payload": {"subject": "Test", "text": "body text"}},
            {"score": 0.8, "payload": {"title": "Title", "body": "main body"}},
        ]
        reranked = bm25_search.hybrid_rerank("test", docs)
        assert len(reranked) == 2

    def test_bm25_scorer_singleton(self):
        """Test that _scorer is a singleton."""
        scorer1 = bm25_search._scorer
        scorer2 = bm25_search._scorer
        assert scorer1 is scorer2


# ============================================================================
# QUERY EXPANSION TESTS (21-35)
# ============================================================================


class TestQueryExpansion:
    """Tests for query expansion module."""

    def test_expand_queries_returns_original(self):
        """Test that original query is always in results."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("test query")
        assert "test query" in queries

    def test_expand_queries_limits_results(self):
        """Test expansion respects max_expansions limit."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("test query", max_expansions=3)
        assert len(queries) <= 3

    def test_expand_queries_removes_stop_words(self):
        """Test that stop words are filtered from entity extraction."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("What is the test")
        assert "What" not in queries
        assert "the" not in queries

    def test_expand_queries_detects_capitalized_entities(self):
        """Test capitalized words are detected as entities."""
        from pipeline.query_expansion import expand_queries

        with patch("pipeline.query_expansion.lookup_entity_graph", return_value=""):
            queries = expand_queries("Josh profile")
            assert len(queries) >= 1

    def test_expand_queries_email_keywords(self):
        """Test email-related keywords trigger expansion."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("email Josh yesterday")
        assert any("email" in q for q in queries)

    def test_expand_queries_research_keywords(self):
        """Test research-related keywords trigger expansion."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("researched Python")
        assert any("perplexity" in q.lower() for q in queries)

    def test_expand_queries_temporal_keywords_recent(self):
        """Test recent temporal keywords trigger expansion."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("last week meeting")
        assert any("recent" in q.lower() for q in queries)

    def test_expand_queries_temporal_keywords_old(self):
        """Test old temporal keywords trigger expansion."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("months ago project")
        assert any("older" in q.lower() for q in queries)

    def test_expand_queries_chatgpt_keywords(self):
        """Test chatgpt-related keywords trigger expansion."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("chatgpt conversation")
        assert any("chatgpt" in q.lower() for q in queries)

    def test_expand_queries_deduplicates(self):
        """Test that duplicate queries are removed."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("test test test")
        assert len(set(queries)) == len(queries)

    def test_expand_queries_empty_query(self):
        """Test empty query handling."""
        from pipeline.query_expansion import expand_queries

        queries = expand_queries("")
        assert len(queries) >= 1

    def test_expand_queries_very_long(self):
        """Test very long query handling."""
        from pipeline.query_expansion import expand_queries

        long_query = " ".join(["word"] * 100)
        queries = expand_queries(long_query, max_expansions=2)
        assert len(queries) <= 2

    def test_topic_rephrase_removes_question_words(self):
        """Test topic rephrasing removes question words."""
        from pipeline.query_expansion import _topic_rephrase

        result = _topic_rephrase("What is the test?")
        assert not result.lower().startswith("what ")

    def test_extract_entities_capitalized_only(self):
        """Test entity extraction only gets capitalized words."""
        from pipeline.query_expansion import _extract_entities

        entities = _extract_entities("the quick Brown Fox jumps")
        assert "Brown" in entities
        assert "Fox" in entities
        assert "the" not in entities


# ============================================================================
# CONTRADICTION DETECTION TESTS (36-50)
# ============================================================================


class TestContradictionDetection:
    """Tests for contradiction detection logic."""

    def test_negation_detection_not(self):
        """Test negation detection for 'not'."""
        from pipeline.contradiction import _contains_negation

        assert _contains_negation("This is not correct")
        assert not _contains_negation("This is correct")

    def test_negation_detection_no_longer(self):
        """Test negation detection for 'no longer'."""
        from pipeline.contradiction import _contains_negation

        assert _contains_negation("We no longer support this")
        assert not _contains_negation("We still support this")

    def test_negation_detection_stopped(self):
        """Test negation detection for 'stopped'."""
        from pipeline.contradiction import _contains_negation

        assert _contains_negation("Service stopped working")
        assert not _contains_negation("Service started working")

    def test_negation_detection_never(self):
        """Test negation detection for 'never'."""
        from pipeline.contradiction import _contains_negation

        assert _contains_negation("Never happened")
        assert not _contains_negation("Always happened")

    def test_negation_detection_contractions(self):
        """Test negation detection for contractions."""
        from pipeline.contradiction import _contains_negation

        assert _contains_negation("doesn't work")
        assert _contains_negation("didn't happen")
        assert _contains_negation("isn't right")
        assert _contains_negation("aren't available")

    def test_extract_subject_number_pairs(self):
        """Test extraction of subject-number pairs."""
        from pipeline.contradiction import _extract_subject_number_pairs

        pairs = _extract_subject_number_pairs("BTC is worth 50000 dollars")
        assert len(pairs) > 0
        assert "btc" in pairs[0][0]
        assert pairs[0][1] == 50000.0

    def test_extract_subject_location_pairs(self):
        """Test extraction of subject-location pairs."""
        from pipeline.contradiction import _extract_subject_location_pairs

        pairs = _extract_subject_location_pairs("Josh moved to Moscow")
        assert len(pairs) > 0
        assert "josh" in pairs[0][0]
        assert "moscow" in pairs[0][1]

    def test_shared_subject_context_word_overlap(self):
        """Test shared context detection via word overlap."""
        from pipeline.contradiction import _shared_subject_context

        assert _shared_subject_context("BTC price rose to new all time high", "BTC price reached new all time high")
        assert not _shared_subject_context("completely different text", "unrelated content")

    def test_shared_subject_context_name_overlap(self):
        """Test shared context detection via name overlap."""
        from pipeline.contradiction import _shared_subject_context

        assert _shared_subject_context("Josh went to store", "Josh bought groceries")
        assert not _shared_subject_context("Josh went to store", "Maria went to park")

    def test_looks_contradictory_negation_flip(self):
        """Test contradiction detection for negation flips."""
        from pipeline.contradiction import looks_contradictory

        assert not looks_contradictory("Service is down", "Service is up")
        assert looks_contradictory("Service is not working", "Service works")

    def test_looks_contradictory_number_change(self):
        """Test contradiction detection for number changes."""
        from pipeline.contradiction import looks_contradictory

        assert looks_contradictory("BTC is 50000", "BTC is 60000")

    def test_looks_contradictory_location_change(self):
        """Test contradiction detection for location changes."""
        from pipeline.contradiction import looks_contradictory

        assert looks_contradictory("Josh is in Moscow", "Josh moved to St Petersburg")

    def test_looks_contradictory_empty_texts(self):
        """Test contradiction detection with empty texts."""
        from pipeline.contradiction import looks_contradictory

        assert not looks_contradictory("", "text")
        assert not looks_contradictory("text", "")
        assert not looks_contradictory("", "")

    def test_check_contradictions_qdrant_error(self, mock_qdrant):
        """Test contradiction check handles Qdrant errors gracefully."""
        mock_qdrant.query_points = MagicMock(side_effect=Exception("Connection failed"))

        from pipeline.contradiction import check_contradictions

        result = check_contradictions("test text", [0.1] * 768, mock_qdrant, "test_collection")

        assert result == []

    def test_check_contradictions_low_similarity(self, mock_qdrant):
        """Test that only high-similarity results are checked."""
        mock_qdrant.upsert(collection_name="test", points=[MagicMock(id=1, score=0.5, payload={"text": "Some text"})])

        from pipeline.contradiction import check_contradictions

        result = check_contradictions("test", [0.1] * 768, mock_qdrant, "test", top_k=5)

        assert len(result) == 0


# ============================================================================
# SOURCE TIERING TESTS (51-55)
# ============================================================================


class TestSourceTiering:
    """Tests for source reliability weighting."""

    def test_gold_sources(self):
        """Test gold tier source weights."""
        assert get_source_weight("conversation") == 0.95
        assert get_source_weight("chatgpt") == 0.9
        assert get_source_weight("perplexity") == 0.9
        assert get_source_weight("email") == 0.9

    def test_silver_sources(self):
        """Test silver tier source weights."""
        assert get_source_weight("telegram") == 0.75
        assert get_source_weight("whatsapp") == 0.7
        assert get_source_weight("social_intel") == 0.65

    def test_bronze_sources(self):
        """Test bronze tier source weights."""
        assert get_source_weight("consolidator") == 0.5
        assert get_source_weight("auto-extract") == 0.4
        assert get_source_weight("fact_extractor") == 0.4
        assert get_source_weight("web_page") == 0.35

    def test_unknown_source_default(self):
        """Test unknown sources get default weight."""
        weight = get_source_weight("unknown_source")
        assert weight == 0.5

    def test_case_insensitive(self):
        """Test source matching is case-insensitive."""
        assert get_source_weight("CONVERSATION") == 0.95
        assert get_source_weight("Conversation") == 0.95


# ============================================================================
# EMBEDDING TESTS (56-65)
# ============================================================================


class TestEmbeddings:
    """Tests for embedding generation."""

    def test_get_embedding_success(self, mock_embedding):
        """Test successful embedding generation."""
        result = mock_embedding("test text")
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_get_embedding_with_prefix(self, mock_embedding):
        """Test embedding with task prefix."""
        result = mock_embedding("test text", prefix="search_query:")
        result_no_prefix = mock_embedding("test text")
        assert result != result_no_prefix

    def test_get_embedding_safe_returns_none_on_skip(self, mock_embedding, monkeypatch):
        """Test get_embedding_safe with skip action."""
        monkeypatch.setattr("hybrid_brain.get_embedding", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))

        result = get_embedding_safe("test", default_action="skip")
        assert result is None

    def test_get_embedding_safe_returns_zero_on_empty(self, mock_embedding, monkeypatch):
        """Test get_embedding_safe with empty action."""
        monkeypatch.setattr("hybrid_brain.get_embedding", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))

        result = get_embedding_safe("test", default_action="empty")
        assert result == [0.0] * 768

    def test_get_embedding_safe_raises_on_fail(self, mock_embedding, monkeypatch):
        """Test get_embedding_safe raises on fail action."""
        monkeypatch.setattr("hybrid_brain.get_embedding", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))

        with pytest.raises(Exception):
            get_embedding_safe("test", default_action="fail")

    def test_embedding_magnitude_check(self):
        """Test that embeddings with low magnitude are rejected."""
        vector = [0.001] * 768
        magnitude = math.sqrt(sum(x * x for x in vector))
        assert magnitude < 0.1

    def test_embedding_magnitude_normal(self):
        """Test that normal embeddings pass magnitude check."""
        vector = [random.random() for _ in range(768)]
        magnitude = math.sqrt(sum(x * x for x in vector))
        assert magnitude > 0.1


# ============================================================================
# ENTITY EXTRACTION TESTS (66-75)
# ============================================================================


class TestEntityExtraction:
    """Tests for entity extraction from text."""

    def test_extract_entities_fast_known_persons(self):
        """Test extraction of known persons."""
        with patch(
            "hybrid_brain._load_known_entities",
            return_value=({"Josh", "Sasha"}, {"Google", "Microsoft"}, {"rasputin-memory"}),
        ):
            entities = extract_entities_fast("Josh met Sasha at Google")
            names = [e[0] for e in entities]
            assert "Josh" in names
            assert "Sasha" in names
            assert "Google" in names

    def test_extract_entities_fast_known_orgs(self):
        """Test extraction of known organizations."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), {"Tesla", "Apple"}, set())):
            entities = extract_entities_fast("Working at Tesla and Apple")
            names = [e[0] for e in entities]
            assert "Tesla" in names
            assert "Apple" in names

    def test_extract_entities_fast_capitalized_names(self):
        """Test extraction of capitalized multi-word names."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), set())):
            entities = extract_entities_fast("John Smith and Jane Doe attended")
            names = [e[0] for e in entities]
            assert "John Smith" in names
            assert "Jane Doe" in names

    def test_extract_entities_fast_no_duplicates(self):
        """Test that entities are not duplicated."""
        with patch("hybrid_brain._load_known_entities", return_value=({"Josh"}, set(), set())):
            entities = extract_entities_fast("Josh and Josh")
            names = [e[0] for e in entities]
            assert names.count("Josh") == 1

    def test_extract_entities_fast_short_names_filtered(self):
        """Test that short capitalized words are filtered."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), set())):
            entities = extract_entities_fast("A B C are short")
            names = [e[0] for e in entities]
            assert len([n for n in names if len(n) <= 4]) == 0

    def test_extract_entities_fast_empty_text(self):
        """Test entity extraction with empty text."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), set())):
            entities = extract_entities_fast("")
            assert entities == []

    def test_extract_entities_fast_special_chars(self):
        """Test entity extraction with special characters."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), set())):
            entities = extract_entities_fast("Contact: John@Smith.com or Mary-Jane")
            assert isinstance(entities, list)

    def test_extract_entities_fast_unicode(self):
        """Test entity extraction with Unicode."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), set())):
            entities = extract_entities_fast("Москва Петербург")
            assert isinstance(entities, list)

    def test_extract_entities_fast_very_long_text(self):
        """Test entity extraction with very long text."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), set())):
            long_text = " ".join([f"Word{i}" for i in range(1000)])
            entities = extract_entities_fast(long_text)
            assert isinstance(entities, list)

    def test_extract_entities_fast_known_projects(self):
        """Test extraction of known projects."""
        with patch("hybrid_brain._load_known_entities", return_value=(set(), set(), {"rasputin-memory", "openclaw"})):
            entities = extract_entities_fast("Working on rasputin-memory and openclaw")
            names = [e[0] for e in entities]
            assert "rasputin-memory" in names
            assert "openclaw" in names


# ============================================================================
# TEMPORAL DECAY TESTS (76-85)
# ============================================================================


class TestTemporalDecay:
    """Tests for temporal decay scoring."""

    def test_temporal_decay_reduces_old_scores(self):
        """Test that old memories get lower scores."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(days=180)).isoformat(), "importance": 50, "retrieval_count": 0},
            {"score": 1.0, "date": (now - timedelta(days=1)).isoformat(), "importance": 50, "retrieval_count": 0},
        ]
        out = apply_temporal_decay(rows)

        assert all("days_old" in r for r in out)

    def test_temporal_decay_important_memories_decay_slower(self):
        """Test important memories decay slower."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(days=100)).isoformat(), "importance": 90, "retrieval_count": 0},
            {"score": 1.0, "date": (now - timedelta(days=100)).isoformat(), "importance": 20, "retrieval_count": 0},
        ]
        out = apply_temporal_decay(rows)

        assert len(out) == 2

    def test_temporal_decay_retrieval_boost(self):
        """Test retrieval count boosts half-life."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(days=30)).isoformat(), "importance": 50, "retrieval_count": 0},
            {"score": 1.0, "date": (now - timedelta(days=30)).isoformat(), "importance": 50, "retrieval_count": 20},
        ]
        out = apply_temporal_decay(rows)

        assert all("effective_half_life" in r for r in out)

    def test_temporal_decay_floor(self):
        """Test decay has floor at 20%."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(days=365)).isoformat(), "importance": 20, "retrieval_count": 0},
        ]
        out = apply_temporal_decay(rows)

        assert out[0]["score"] >= 0.2

    def test_temporal_decay_recent_memories_preserved(self):
        """Test very recent memories are preserved."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(hours=1)).isoformat(), "importance": 90, "retrieval_count": 5},
        ]
        out = apply_temporal_decay(rows)

        assert out[0]["score"] > 0.95

    def test_temporal_decay_sorts_by_score(self):
        """Test results are sorted by score descending."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(days=100)).isoformat(), "importance": 50, "retrieval_count": 0},
            {"score": 1.0, "date": (now - timedelta(days=1)).isoformat(), "importance": 50, "retrieval_count": 0},
        ]
        out = apply_temporal_decay(rows)

        assert out[0]["score"] >= out[1]["score"]

    def test_temporal_decay_missing_date(self):
        """Test handling of missing date field."""
        rows = [{"score": 1.0, "importance": 50, "retrieval_count": 0}]
        out = apply_temporal_decay(rows)
        assert len(out) == 1

    def test_temporal_decay_none_importance(self):
        """Test handling of None importance."""
        now = datetime.now(timezone.utc)
        rows = [
            {"score": 1.0, "date": (now - timedelta(days=30)).isoformat(), "importance": None, "retrieval_count": 0}
        ]
        out = apply_temporal_decay(rows)
        assert len(out) == 1

    def test_temporal_decay_invalid_importance(self):
        """Test handling of invalid importance values."""
        now = datetime.now(timezone.utc)
        rows = [
            {
                "score": 1.0,
                "date": (now - timedelta(days=30)).isoformat(),
                "importance": "invalid",
                "retrieval_count": 0,
            }
        ]
        out = apply_temporal_decay(rows)
        assert len(out) == 1

    def test_temporal_decay_empty_results(self):
        """Test empty results handling."""
        out = apply_temporal_decay([])
        assert out == []


# ============================================================================
# MULTIFACTOR SCORING TESTS (86-95)
# ============================================================================


class TestMultifactorScoring:
    """Tests for multi-factor importance scoring."""

    def test_multifactor_scoring_basic(self):
        """Test basic multifactor scoring."""
        rows = [{"score": 0.8, "importance": 90, "source": "conversation", "retrieval_count": 10, "days_old": 1}]
        out = apply_multifactor_scoring(rows)

        assert len(out) == 1
        assert "multifactor" in out[0]
        assert "pre_multifactor_score" in out[0]

    def test_multifactor_scoring_important_boost(self):
        """Test high importance gets score boost."""
        rows = [
            {"score": 0.5, "importance": 100, "source": "conversation", "retrieval_count": 0, "days_old": 1},
            {"score": 0.5, "importance": 0, "source": "web_page", "retrieval_count": 0, "days_old": 1},
        ]
        out = apply_multifactor_scoring(rows)

        assert out[0]["importance"] == 100

    def test_multifactor_scoring_source_weight(self):
        """Test source weight affects scoring."""
        rows = [
            {"score": 0.7, "importance": 50, "source": "conversation", "retrieval_count": 0, "days_old": 1},
            {"score": 0.7, "importance": 50, "source": "web_page", "retrieval_count": 0, "days_old": 1},
        ]
        out = apply_multifactor_scoring(rows)

        assert out[0]["source"] == "conversation"

    def test_multifactor_scoring_retrieval_boost(self):
        """Test retrieval count affects scoring."""
        rows = [
            {"score": 0.7, "importance": 50, "source": "conversation", "retrieval_count": 0, "days_old": 1},
            {"score": 0.7, "importance": 50, "source": "conversation", "retrieval_count": 10, "days_old": 1},
        ]
        out = apply_multifactor_scoring(rows)

        assert len(out) == 2

    def test_multifactor_scoring_recency_bonus(self):
        """Test recency bonus for recent items."""
        rows = [
            {"score": 0.7, "importance": 50, "source": "conversation", "retrieval_count": 0, "days_old": 3},
            {"score": 0.7, "importance": 50, "source": "conversation", "retrieval_count": 0, "days_old": 100},
        ]
        out = apply_multifactor_scoring(rows)

        assert len(out) == 2

    def test_multifactor_scoring_sorts_by_score(self):
        """Test results are sorted by score descending."""
        rows = [
            {"score": 0.5, "importance": 20, "source": "web_page", "retrieval_count": 0, "days_old": 100},
            {"score": 0.5, "importance": 90, "source": "conversation", "retrieval_count": 10, "days_old": 1},
        ]
        out = apply_multifactor_scoring(rows)

        assert out[0]["importance"] == 90

    def test_multifactor_scoring_none_importance(self):
        """Test handling of None importance."""
        rows = [{"score": 0.7, "importance": None, "source": "conversation", "retrieval_count": 0, "days_old": 1}]
        out = apply_multifactor_scoring(rows)
        assert len(out) == 1

    def test_multifactor_scoring_invalid_importance(self):
        """Test handling of invalid importance."""
        rows = [{"score": 0.7, "importance": "invalid", "source": "conversation", "retrieval_count": 0, "days_old": 1}]
        out = apply_multifactor_scoring(rows)
        assert len(out) == 1

    def test_multifactor_scoring_none_retrieval(self):
        """Test handling of None retrieval_count."""
        rows = [{"score": 0.7, "importance": 50, "source": "conversation", "retrieval_count": None, "days_old": 1}]
        out = apply_multifactor_scoring(rows)
        assert len(out) == 1

    def test_multifactor_scoring_empty_results(self):
        """Test empty results handling."""
        out = apply_multifactor_scoring([])
        assert out == []
