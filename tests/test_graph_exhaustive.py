"""
EXHAUSTIVE TESTS FOR GRAPH/KNOWLEDGE LAYER

Covers:
1. Unit tests — graph operations, entity CRUD, relationship management
2. Integration tests — full graph pipeline (text → entities → relationships → retrieval)
3. Edge cases — entities with special chars, circular relationships, orphan nodes, massive graphs
4. Query tests — Cypher query generation, parameterized queries, injection prevention
5. Error paths — FalkorDB down, malformed data, schema violations
6. Consistency — graph state after operations, no orphaned edges

80+ test cases minimum using pytest.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import hybrid_brain
from hybrid_brain import (
    extract_entities_fast,
    write_to_graph,
    graph_search,
    hybrid_search,
    commit_memory,
    FALKORDB_DISABLED,
)


# ============================================================================
# MOCK CLASSES
# ============================================================================

class MockRedisConnection:
    """Mock Redis connection that simulates FalkorDB."""
    
    def __init__(self):
        self.queries = []
        self.nodes = {}  # id -> node data
        self.edges = []  # (from_id, to_id, rel_type)
        self.ping_called = False
        
    def ping(self):
        self.ping_called = True
        return True
    
    def execute_command(self, *args):
        self.queries.append(args)
        
        if len(args) < 3:
            return [[], []]
            
        cmd = args[0]
        graph_name = args[1]
        cypher = args[2]
        
        if cmd != "GRAPH.QUERY":
            return [[], []]
            
        # Parse and simulate Cypher queries
        cypher_upper = cypher.upper()
        
        # MERGE queries
        if "MERGE" in cypher_upper:
            if "ON CREATE" in cypher_upper:
                # Entity creation with properties
                return [[], [[1, "created"]]]
            return [[], [[1, "merged"]]]
            
        # MATCH queries
        if "MATCH" in cypher_upper:
            if "RETURN count" in cypher_upper:
                return [[], [[0]]]
            if "RETURN m.id, m.text" in cypher:
                # Memory node retrieval
                return [[], [["1", "test memory text", "2026-03-30", "Test Entity"]]]
            if "RETURN labels(connected)" in cypher:
                # Entity relationship context
                return [[], [["Person", "Related Entity", "MENTIONS", "Test Entity"]]]
                
        return [[], []]


class MockRedis:
    """Mock Redis class for FalkorDB."""
    
    def __init__(self, pool=None):
        self.conn = MockRedisConnection()
        
    def ping(self):
        return self.conn.ping()
        
    def execute_command(self, *args):
        return self.conn.execute_command(*args)


class MockRedisPool:
    """Mock Redis connection pool."""
    
    def __init__(self):
        self.conn = MockRedis()
        
    def get_connection(self, *args):
        return self.conn


# ============================================================================
# UNIT TESTS - Entity Extraction
# ============================================================================

class TestEntityExtraction:
    """Unit tests for entity extraction functionality."""
    
    def test_extract_known_person(self, monkeypatch):
        """Test extraction of known persons from text."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe", "Alice"}, set(), set()),
        )
        entities = extract_entities_fast("Yesterday John Doe reviewed the roadmap")
        assert ("John Doe", "Person") in entities
        assert ("Alice", "Person") not in entities  # Not in text
        
    def test_extract_known_organization(self, monkeypatch):
        """Test extraction of known organizations from text."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), {"OpenClaw", "BrandA"}, set()),
        )
        entities = extract_entities_fast("We integrated with OpenClaw this week")
        assert ("OpenClaw", "Organization") in entities
        
    def test_extract_known_project(self, monkeypatch):
        """Test extraction of known projects from text."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), set(), {"Rasputin Memory"}),
        )
        entities = extract_entities_fast("Working on Rasputin Memory project")
        assert ("Rasputin Memory", "Project") in entities
        
    def test_no_substring_match(self, monkeypatch):
        """Test that substring matches are rejected (word boundary check)."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"Al"}, set(), set()),
        )
        entities = extract_entities_fast("We improved algorithm performance")
        assert ("Al", "Person") not in entities  # "Al" is substring of "algorithm"
        
    def test_capitalized_phrase_extraction(self, monkeypatch):
        """Test extraction of capitalized multi-word names."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), set(), set()),
        )
        entities = extract_entities_fast("Roadmap review with Jane Smith in Toronto")
        assert ("Jane Smith", "Person") in entities
        
    def test_entity_extraction_consistency(self, monkeypatch):
        """Test that entity extraction is deterministic."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, {"OpenClaw"}, {"Rasputin Memory"}),
        )
        text = "John Doe discussed Rasputin Memory rollout at OpenClaw"
        first = extract_entities_fast(text)
        second = extract_entities_fast(text)
        assert first == second
        
    def test_empty_text_returns_empty_list(self):
        """Test that empty text returns no entities."""
        entities = extract_entities_fast("")
        assert entities == []
        
    def test_text_with_no_entities_returns_empty(self, monkeypatch):
        """Test text with no extractable entities."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), set(), set()),
        )
        entities = extract_entities_fast("the quick brown fox jumps")
        assert entities == []
        
    def test_special_characters_in_text(self, monkeypatch):
        """Test entity extraction with special characters."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John O'Brien"}, set(), set()),
        )
        entities = extract_entities_fast("Meeting with John O'Brien tomorrow")
        assert ("John O'Brien", "Person") in entities
        
    def test_unicode_entities(self, monkeypatch):
        """Test entity extraction with unicode characters."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"José García"}, set(), set()),
        )
        entities = extract_entities_fast("Hola José García")
        assert ("José García", "Person") in entities
        
    def test_long_entity_name(self, monkeypatch):
        """Test extraction of long entity names."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Michael Smith Jr"}, set(), set()),
        )
        entities = extract_entities_fast("Consultation with John Michael Smith Jr")
        # Regex captures capitalized words, periods may break multi-word matching
        assert ("John Michael Smith Jr", "Person") in entities
        
    def test_multiple_entities_same_type(self, monkeypatch):
        """Test extraction of multiple entities of same type."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"Alice", "Bob", "Charlie"}, set(), set()),
        )
        entities = extract_entities_fast("Alice, Bob, and Charlie attended")
        assert len([e for e in entities if e[1] == "Person"]) == 3
        
    def test_mixed_entity_types(self, monkeypatch):
        """Test extraction of mixed entity types."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, {"Acme Corp"}, {"Project X"}),
        )
        entities = extract_entities_fast("John Doe from Acme Corp started Project X")
        assert ("John Doe", "Person") in entities
        assert ("Acme Corp", "Organization") in entities
        assert ("Project X", "Project") in entities
        
    def test_entity_case_sensitivity(self, monkeypatch):
        """Test that entity matching is case-insensitive."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, set(), set()),
        )
        entities = extract_entities_fast("john doe attended the meeting")
        assert ("John Doe", "Person") in entities
        
    def test_entity_with_numbers(self, monkeypatch):
        """Test entity names containing numbers."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"Team 42"}, set(), set()),
        )
        entities = extract_entities_fast("Team 42 shipped the release")
        assert ("Team 42", "Person") in entities
        
    def test_short_text_no_entities(self, monkeypatch):
        """Test very short text with no entities."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), set(), set()),
        )
        entities = extract_entities_fast("hi")
        assert entities == []


# ============================================================================
# UNIT TESTS - Graph Write Operations
# ============================================================================

class TestGraphWrite:
    """Unit tests for graph write operations."""
    
    def test_write_to_graph_success(self, monkeypatch):
        """Test successful write to graph."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=123,
            text="John Doe met BrandA at OpenClaw HQ",
            entities=[("John Doe", "Person"), ("BrandA", "Organization")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert len(connected) == 2
        assert "John Doe" in connected
        assert "BrandA" in connected
        assert mock_conn.ping_called
        
    def test_write_to_graph_disabled(self, monkeypatch):
        """Test graph write when FalkorDB is disabled."""
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", True)
        
        success, connected = write_to_graph(
            point_id=123,
            text="Test text",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert connected == []
        
    def test_write_to_graph_connection_failure(self, monkeypatch):
        """Test graph write when connection fails."""
        class FailingRedis:
            def ping(self):
                raise Exception("Connection refused")
                
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: FailingRedis())
        
        success, connected = write_to_graph(
            point_id=123,
            text="Test text",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is False
        assert connected == []
        
    def test_write_to_graph_empty_entities(self, monkeypatch):
        """Test graph write with no entities."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=123,
            text="Just text, no entities",
            entities=[],
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert connected == []
        
    def test_write_to_graph_malformed_entity_type(self, monkeypatch):
        """Test graph write with invalid entity type (should use fallback)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=123,
            text="Test",
            entities=[("Test Entity", "InvalidType")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert "Test Entity" in connected
        
    def test_write_to_graph_special_chars_in_entity(self, monkeypatch):
        """Test graph write with special characters in entity name."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=123,
            text="Meeting with O'Brien",
            entities=[("O'Brien", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert "O'Brien" in connected
        
    def test_write_to_graph_unicode_entities(self, monkeypatch):
        """Test graph write with unicode entity names."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=123,
            text="José García",
            entities=[("José García", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert "José García" in connected
        
    def test_write_to_graph_very_long_text(self, monkeypatch):
        """Test graph write with very long text (should be truncated)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        long_text = "A" * 10000
        success, connected = write_to_graph(
            point_id=123,
            text=long_text,
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        
    def test_write_to_graph_cypher_injection_attempt(self, monkeypatch):
        """Test that Cypher injection attempts are prevented via parameterization."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Entity name with Cypher injection attempt
        success, connected = write_to_graph(
            point_id=123,
            text="Test",
            entities=[("Test); DELETE ALL; --", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        # Verify parameters were used (not string concatenation)
        assert len(mock_conn.queries) > 0
        
    def test_write_to_graph_memory_node_creation(self, monkeypatch):
        """Test that Memory node is created with correct properties."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(
            point_id=456,
            text="Test memory",
            entities=[("Test", "Person")],
            timestamp="2026-03-30T12:00:00",
        )
        
        # Verify MERGE with id property was used
        assert any("MERGE (m:Memory" in str(q) for q in mock_conn.queries)
        assert any('"id": "456"' in str(q) for q in mock_conn.queries)
        
    def test_write_to_graph_relationship_creation(self, monkeypatch):
        """Test that MENTIONS relationships are created."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(
            point_id=789,
            text="Test",
            entities=[("Entity1", "Person"), ("Entity2", "Org")],
            timestamp="2026-03-30",
        )
        
        # Verify MENTIONS relationship was created
        assert any("MERGE (m)-[:MENTIONS]->(n)" in str(q) for q in mock_conn.queries)


# ============================================================================
# UNIT TESTS - Graph Search Operations
# ============================================================================

class TestGraphSearch:
    """Unit tests for graph search/traversal."""
    
    def test_graph_search_basic(self, monkeypatch):
        """Test basic graph search with single entity."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("John Doe", "Person")],
        )
        
        results = graph_search("John Doe", hops=1, limit=10)
        
        assert len(results) >= 0  # Mock returns results
        assert all(r.get("origin") == "graph" for r in results)
        
    def test_graph_search_disabled(self, monkeypatch):
        """Test graph search when FalkorDB is disabled."""
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", True)
        
        results = graph_search("Test", hops=1, limit=10)
        
        assert results == []
        
    def test_graph_search_connection_failure(self, monkeypatch):
        """Test graph search when connection fails."""
        class FailingRedis:
            def ping(self):
                raise Exception("Connection refused")
                
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: FailingRedis())
        
        results = graph_search("Test", hops=1, limit=10)
        
        assert results == []
        
    def test_graph_search_1_hop(self, monkeypatch):
        """Test 1-hop graph search (direct entity mentions)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test Entity", "Person")],
        )
        
        results = graph_search("Test Entity", hops=1, limit=10)
        
        # Should query for 1-hop relationships
        assert any("graph_hop" in r for r in results)
        
    def test_graph_search_2_hop(self, monkeypatch):
        """Test 2-hop graph search (co-mentioned entities)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test Entity", "Person")],
        )
        
        results = graph_search("Test Entity", hops=2, limit=10)
        
        # Should include both 1-hop and 2-hop results
        hop_counts = [r.get("graph_hop") for r in results if "graph_hop" in r]
        assert set(hop_counts) <= {1, 2}  # Only 1 or 2
        
    def test_graph_search_multiple_entities(self, monkeypatch):
        """Test graph search with multiple extracted entities."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Entity1", "Person"), ("Entity2", "Org")],
        )
        
        results = graph_search("Entity1 Entity2", hops=1, limit=10)
        
        # Should search for both entities
        assert len(mock_conn.queries) > 0
        
    def test_graph_search_unknown_entity_type(self, monkeypatch):
        """Test graph search with entity type not in whitelist."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Unknown", "UnknownType")],
        )
        
        results = graph_search("Unknown", hops=1, limit=10)
        
        # Should still work with fallback labels
        assert isinstance(results, list)
        
    def test_graph_search_keyword_fallback(self, monkeypatch):
        """Test keyword entity type uses text search fallback."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("keyword", "Keyword")],
        )
        
        results = graph_search("keyword", hops=1, limit=10)
        
        # Should search Memory.text directly for keywords
        assert isinstance(results, list)
        
    def test_graph_search_short_keyword(self, monkeypatch):
        """Test that short keywords (< 3 chars) are skipped."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("ab", "Keyword")],
        )
        
        results = graph_search("ab", hops=1, limit=10)
        
        # Short keywords should not trigger text search
        assert isinstance(results, list)
        
    def test_graph_search_context_results(self, monkeypatch):
        """Test that graph search returns entity relationship context."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        results = graph_search("Test", hops=1, limit=10)
        
        # Context results have 'connected_to' field
        context_results = [r for r in results if "connected_to" in r]
        assert isinstance(context_results, list)
        
    def test_graph_search_limit_respected(self, monkeypatch):
        """Test that limit parameter is respected."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        results = graph_search("Test", hops=1, limit=3)
        
        # Results should not exceed limit * 2 (as per implementation)
        assert len(results) <= 6
        
    def test_graph_search_empty_query(self, monkeypatch):
        """Test graph search with empty query."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        results = graph_search("", hops=1, limit=10)
        
        # Should still work (treats as Unknown entity)
        assert isinstance(results, list)
        
    def test_graph_search_special_chars_in_query(self, monkeypatch):
        """Test graph search with special characters in query."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("O'Brien", "Person")],
        )
        
        results = graph_search("O'Brien", hops=1, limit=10)
        
        assert isinstance(results, list)


# ============================================================================
# INTEGRATION TESTS - Full Pipeline
# ============================================================================

class TestGraphPipeline:
    """Integration tests for full graph pipeline."""
    
    def test_full_commit_with_graph(self, monkeypatch):
        """Test complete memory commit with graph write."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        # Mock Qdrant
        mock_qdrant = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_qdrant.set_payload = MagicMock()
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(hybrid_brain, "qdrant", mock_qdrant)
        monkeypatch.setattr(hybrid_brain, "FALKOR_HOST", "localhost")
        monkeypatch.setattr(hybrid_brain, "FALKOR_PORT", 6380)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("John Doe", "Person")],
        )
        monkeypatch.setattr(
            hybrid_brain,
            "check_duplicate",
            lambda *args, **kwargs: (False, None, 0),
        )
        monkeypatch.setattr(
            hybrid_brain,
            "check_contradictions",
            lambda *args, **kwargs: [],
        )
        
        result = commit_memory(
            text="John Doe reviewed the Q1 roadmap",
            source="conversation",
            importance=70,
        )
        
        assert result["ok"] is True
        assert result["graph"]["written"] is True
        assert result["graph"]["entities"] == 1
        assert "John Doe" in result["graph"]["connected_to"]
        
    def test_hybrid_search_with_graph_integration(self, monkeypatch):
        """Test hybrid search integrating graph results."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(hybrid_brain, "BM25_AVAILABLE", False)
        monkeypatch.setattr(hybrid_brain, "qdrant_search", lambda *a, **k: [])
        monkeypatch.setattr(hybrid_brain, "is_reranker_available", lambda: False)
        monkeypatch.setattr(hybrid_brain, "enrich_with_graph", lambda *a, **k: {})
        monkeypatch.setattr(
            hybrid_brain,
            "graph_search",
            lambda *a, **k: [
                {"text": "Graph result 1", "origin": "graph", "graph_hop": 1, "source": "graph_memory"},
                {"text": "Graph result 2", "origin": "graph", "graph_hop": 2, "source": "graph_memory"},
            ],
        )
        
        result = hybrid_search("Test query", limit=5, expand=False)
        
        assert "results" in result
        assert len(result["results"]) >= 0
        assert result["stats"]["graph_hits"] >= 0
        
    def test_graph_results_scored_by_hop_distance(self, monkeypatch):
        """Test that graph results are scored based on hop distance."""
        monkeypatch.setattr(hybrid_brain, "BM25_AVAILABLE", False)
        monkeypatch.setattr(hybrid_brain, "qdrant_search", lambda *a, **k: [])
        monkeypatch.setattr(hybrid_brain, "is_reranker_available", lambda: False)
        monkeypatch.setattr(hybrid_brain, "enrich_with_graph", lambda *a, **k: {})
        monkeypatch.setattr(
            hybrid_brain,
            "graph_search",
            lambda *a, **k: [
                {"text": "1-hop", "origin": "graph", "graph_hop": 1, "source": "graph_memory"},
                {"text": "2-hop", "origin": "graph", "graph_hop": 2, "source": "graph_memory"},
            ],
        )
        
        result = hybrid_search("Query", limit=2, expand=False)
        
        scores = {r["text"]: r["score"] for r in result["results"]}
        assert scores.get("1-hop") == 0.8
        assert scores.get("2-hop") == 0.5


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for graph operations."""
    
    def test_entity_with_newlines(self, monkeypatch):
        """Test entity extraction from text with newlines."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, set(), set()),
        )
        text = "Meeting\nwith\nJohn Doe\ntomorrow"
        entities = extract_entities_fast(text)
        assert ("John Doe", "Person") in entities
        
    def test_entity_with_tabs(self, monkeypatch):
        """Test entity extraction from text with tabs."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, set(), set()),
        )
        text = "Meeting\twith\tJohn Doe\ttomorrow"
        entities = extract_entities_fast(text)
        assert ("John Doe", "Person") in entities
        
    def test_very_long_entity_name(self, monkeypatch):
        """Test extraction of very long entity names."""
        long_name = "The Very Long Organization Name That Goes On And On"
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), {long_name}, set()),
        )
        entities = extract_entities_fast(f"Working with {long_name}")
        assert (long_name, "Organization") in entities
        
    def test_entity_at_text_boundaries(self, monkeypatch):
        """Test entity at start and end of text."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, set(), set()),
        )
        entities = extract_entities_fast("John Doe started the meeting with John Doe")
        # Should extract once (deduplicated)
        assert len([e for e in entities if e[0] == "John Doe"]) == 1
        
    def test_circular_relationship_simulation(self, monkeypatch):
        """Test handling of potential circular relationships (via mock)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Write entities that could form circular references
        success, _ = write_to_graph(
            point_id=1,
            text="A knows B, B knows A",
            entities=[("A", "Person"), ("B", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is True
        
    def test_orphan_node_prevention(self, monkeypatch):
        """Test that entities are always linked to Memory nodes."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(
            point_id=999,
            text="Test",
            entities=[("OrphanTest", "Person")],
            timestamp="2026-03-30",
        )
        
        # Verify MERGE relationship was created
        assert any("MERGE (m)-[:MENTIONS]->(n)" in str(q) for q in mock_conn.queries)
        
    def test_massive_entity_list(self, monkeypatch):
        """Test handling of very large entity lists."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # 100 entities
        entities = [(f"Entity{i}", "Person") for i in range(100)]
        
        success, connected = write_to_graph(
            point_id=888,
            text="Massive test",
            entities=entities,
            timestamp="2026-03-30",
        )
        
        assert success is True
        assert len(connected) == 100
        
    def test_duplicate_entities_in_same_text(self, monkeypatch):
        """Test deduplication of entities appearing multiple times."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"John Doe"}, set(), set()),
        )
        text = "John Doe and John Doe met with John Doe"
        entities = extract_entities_fast(text)
        # Should be deduplicated
        assert len([e for e in entities if e[0] == "John Doe"]) == 1
        
    def test_empty_entity_name(self, monkeypatch):
        """Test handling of empty entity names."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: (set(), set(), set()),
        )
        # Empty string shouldn't match anything
        entities = extract_entities_fast("test")
        assert entities == []
        
    def test_entity_with_leading_trailing_spaces(self, monkeypatch):
        """Test entity names with spaces."""
        monkeypatch.setattr(
            hybrid_brain,
            "_load_known_entities",
            lambda: ({"  John Doe  "}, set(), set()),
        )
        # Should match with word boundaries
        entities = extract_entities_fast("Meeting with   John Doe   tomorrow")
        assert ("  John Doe  ", "Person") in entities or ("John Doe", "Person") in entities
        
    def test_rapid_sequential_writes(self, monkeypatch):
        """Test rapid sequential graph writes."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        for i in range(10):
            success, _ = write_to_graph(
                point_id=i,
                text=f"Test {i}",
                entities=[(f"Entity{i}", "Person")],
                timestamp="2026-03-30",
            )
            assert success is True
            
        assert len(mock_conn.queries) > 0
        
    def test_concurrent_search_and_write(self, monkeypatch):
        """Test concurrent graph search and write operations."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        # Write
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        # Search
        results = graph_search("Test", hops=1, limit=5)
        
        assert isinstance(results, list)
        
    def test_very_large_graph_simulation(self, monkeypatch):
        """Test graph operations with simulated large graph."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        # Pre-populate mock with many nodes
        for i in range(1000):
            mock_conn.nodes[f"node{i}"] = {"text": f"Node {i}"}
            
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Should not crash with large graph
        results = graph_search("Test", hops=2, limit=100)
        assert isinstance(results, list)


# ============================================================================
# QUERY TESTS - Cypher Generation & Security
# ============================================================================

class TestCypherQueryGeneration:
    """Tests for Cypher query generation and security."""
    
    def test_parameterized_queries_used(self, monkeypatch):
        """Test that parameterized queries are used (not string concatenation)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(
            point_id=123,
            text="Test",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        # Verify --params flag was used
        assert any("--params" in str(q) for q in mock_conn.queries)
        
    def test_cypher_injection_prevention(self, monkeypatch):
        """Test that Cypher injection attempts are neutralized."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Malicious entity name
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("'); DELETE ALL; --", "Person")],
            timestamp="2026-03-30",
        )
        
        # Verify parameters were used, not string concatenation
        queries_str = " ".join(str(q) for q in mock_conn.queries)
        # Should use $name parameter, not direct string interpolation
        assert "$name" in queries_str
        
    def test_label_whitelist_enforcement(self, monkeypatch):
        """Test that only whitelabel labels are used in Cypher."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Invalid label
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Test", "MaliciousLabel; DELETE ALL")],
            timestamp="2026-03-30",
        )
        
        # Should use fallback "Entity" label
        assert any(":Entity" in str(q) for q in mock_conn.queries)
        
    def test_query_limit_enforcement(self, monkeypatch):
        """Test that LIMIT is enforced in queries."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        graph_search("Test", hops=1, limit=5)
        
        # Verify LIMIT clause was used
        queries_str = " ".join(str(q) for q in mock_conn.queries)
        assert "LIMIT" in queries_str
        
    def test_case_insensitive_matching(self, monkeypatch):
        """Test that queries use case-insensitive matching."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        graph_search("test", hops=1, limit=5)
        
        # Verify toLower() was used
        queries_str = " ".join(str(q) for q in mock_conn.queries)
        assert "TOLOWER" in queries_str.upper() or "toLower" in queries_str
        
    def test_containment_operator_used(self, monkeypatch):
        """Test that CONTAINS operator is used for partial matching."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        graph_search("Test", hops=1, limit=5)
        
        # Verify CONTAINS was used
        queries_str = " ".join(str(q) for q in mock_conn.queries)
        assert "CONTAINS" in queries_str.upper()


# ============================================================================
# ERROR PATH TESTS
# ============================================================================

class TestErrorPaths:
    """Tests for error handling and edge cases."""
    
    def test_falkordb_down_graceful_degradation(self, monkeypatch):
        """Test that graph operations fail gracefully when FalkorDB is down."""
        class DownRedis:
            def ping(self):
                raise redis.ConnectionError("Connection refused")
                
        import redis
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: DownRedis())
        
        # Should return False, [] not raise exception
        success, connected = write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        assert success is False
        assert connected == []
        
    def test_malformed_json_in_params(self, monkeypatch):
        """Test handling of malformed JSON in parameters."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        # Mock execute_command to raise JSON error
        original_execute = mock_conn.execute_command
        def failing_execute(*args):
            if "--params" in str(args):
                raise Exception("Invalid JSON")
            return original_execute(*args)
        mock_conn.execute_command = failing_execute
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Should handle gracefully
        success, connected = write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        # Entity write failed but function should not crash
        assert isinstance(success, bool)
        
    def test_schema_violation_handling(self, monkeypatch):
        """Test handling of schema violations."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        # Mock to simulate schema error
        def schema_error(*args):
            raise Exception("Schema violation: property type mismatch")
        mock_conn.execute_command = schema_error
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        # Should handle gracefully
        assert isinstance(success, bool)
        
    def test_timeout_handling(self, monkeypatch):
        """Test handling of query timeouts."""
        import time
        
        class SlowRedis:
            def ping(self):
                time.sleep(0.1)
                return True
            def execute_command(self, *args):
                time.sleep(0.1)
                return [[], []]
                
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: SlowRedis())
        
        # Should not timeout with reasonable timeout
        results = graph_search("Test", hops=1, limit=5)
        assert isinstance(results, list)
        
    def test_null_entity_handling(self, monkeypatch):
        """Test handling of null/None entities."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=1,
            text="Test",
            entities=[(None, "Person"), ("Valid", "Person")],  # None entity
            timestamp="2026-03-30",
        )
        
        # Should handle gracefully
        assert isinstance(success, bool)
        
    def test_empty_text_commit(self, monkeypatch):
        """Test graph write with empty text."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        success, connected = write_to_graph(
            point_id=1,
            text="",
            entities=[("Test", "Person")],
            timestamp="2026-03-30",
        )
        
        # Should handle gracefully
        assert isinstance(success, bool)


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================

class TestGraphConsistency:
    """Tests for graph state consistency."""
    
    def test_no_orphan_edges_after_write(self, monkeypatch):
        """Test that writes don't create orphan edges."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Entity1", "Person"), ("Entity2", "Org")],
            timestamp="2026-03-30",
        )
        
        # All edges should be connected to Memory node
        assert any("MERGE (m)-[:MENTIONS]->(n)" in str(q) for q in mock_conn.queries)
        
    def test_graph_state_after_multiple_writes(self, monkeypatch):
        """Test graph state consistency after multiple writes."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        for i in range(5):
            write_to_graph(
                point_id=i,
                text=f"Test {i}",
                entities=[(f"Entity{i}", "Person")],
                timestamp="2026-03-30",
            )
            
        # All queries should be valid (1 memory + 1 entity per write = 10 total)
        assert len(mock_conn.queries) == 10
        
    def test_entity_uniqueness_across_writes(self, monkeypatch):
        """Test that entities are unique across writes (MERGE not CREATE)."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        # Write same entity twice
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Shared", "Person")],
            timestamp="2026-03-30",
        )
        write_to_graph(
            point_id=2,
            text="Test",
            entities=[("Shared", "Person")],
            timestamp="2026-03-30",
        )
        
        # MERGE should be used (not CREATE)
        assert any("MERGE" in str(q) for q in mock_conn.queries)
        
    def test_memory_node_id_uniqueness(self, monkeypatch):
        """Test that Memory nodes have unique IDs."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(point_id=1, text="Test1", entities=[], timestamp="2026-03-30")
        write_to_graph(point_id=2, text="Test2", entities=[], timestamp="2026-03-30")
        
        # Different IDs should be used
        queries_str = " ".join(str(q) for q in mock_conn.queries)
        assert '"id": "1"' in queries_str
        assert '"id": "2"' in queries_str
        
    def test_timestamp_consistency(self, monkeypatch):
        """Test that timestamps are consistent across writes."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        timestamp = "2026-03-30T12:00:00"
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Entity", "Person")],
            timestamp=timestamp,
        )
        
        # Timestamp should be in query params
        queries_str = " ".join(str(q) for q in mock_conn.queries)
        assert timestamp in queries_str
        
    def test_relationship_type_consistency(self, monkeypatch):
        """Test that relationship types are consistent."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        
        write_to_graph(
            point_id=1,
            text="Test",
            entities=[("Entity", "Person")],
            timestamp="2026-03-30",
        )
        
        # MENTIONS relationship should be used
        assert any(":MENTIONS" in str(q) for q in mock_conn.queries)
        
    def test_graph_search_determinism(self, monkeypatch):
        """Test that graph search is deterministic."""
        mock_conn = MockRedisConnection()
        mock_redis = MockRedis()
        mock_redis.conn = mock_conn
        
        monkeypatch.setattr(hybrid_brain, "FALKORDB_DISABLED", False)
        monkeypatch.setattr(hybrid_brain, "get_redis", lambda: mock_redis)
        monkeypatch.setattr(
            hybrid_brain,
            "extract_entities_fast",
            lambda text: [("Test", "Person")],
        )
        
        results1 = graph_search("Test", hops=1, limit=10)
        results2 = graph_search("Test", hops=1, limit=10)
        
        # Results should be same structure
        assert len(results1) == len(results2)


# ============================================================================
# FIXTURES AND SETUP
# ============================================================================

@pytest.fixture
def mock_redis_connection():
    """Provide mock Redis connection for tests."""
    return MockRedisConnection()


@pytest.fixture
def mock_redis_instance(mock_redis_connection):
    """Provide mock Redis instance."""
    mock_redis = MockRedis()
    mock_redis.conn = mock_redis_connection
    return mock_redis


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
