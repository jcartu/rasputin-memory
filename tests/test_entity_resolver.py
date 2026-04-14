from __future__ import annotations

from typing import Any

import pytest

from brain import entity_resolver
from brain.entity_resolver import EntityResolver


class _FalkorMock:
    def __init__(self, entity_rows: list[list[str]] | None = None) -> None:
        self._entity_rows = entity_rows or []

    def execute_command(self, *args: Any) -> Any:
        if len(args) >= 3 and args[0] == "GRAPH.QUERY":
            query = str(args[2])
            if "n.canonical" in query:
                return [[], self._entity_rows]
            return [[], []]
        return "OK"


@pytest.fixture(autouse=True)
def _enable_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entity_resolver, "ENTITY_RESOLVER_ENABLED", True)
    monkeypatch.setattr(entity_resolver, "FUZZY_THRESHOLD", 0.85)


@pytest.fixture()
def resolver() -> EntityResolver:
    return EntityResolver()


def _seed(
    resolver: EntityResolver,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[list[str]],
) -> None:
    mock = _FalkorMock(rows)
    monkeypatch.setattr("brain._state.get_falkordb", lambda: mock)
    monkeypatch.setattr("brain._state.GRAPH_NAME", "test_brain")
    resolver._load_cache("test_brain")


def test_exact_match(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["Alice", "Alice", "Person"]])
    result = resolver.resolve([("Alice", "Person")], "Alice went home")
    assert result == [("Alice", "Alice", "Person")]


def test_case_insensitive(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["Alice", "Alice", "Person"]])
    result = resolver.resolve([("alice", "Person")], "alice went home")
    assert result == [("alice", "Alice", "Person")]


def test_substring_match(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["my friend Alice", "my friend Alice", "Person"]])
    result = resolver.resolve([("Alice", "Person")], "Alice went home")
    assert result[0][1] == "my friend Alice"


def test_substring_reverse(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["Alice", "Alice", "Person"]])
    result = resolver.resolve([("my friend Alice", "Person")], "my friend Alice went home")
    assert result[0][1] == "Alice"


def test_fuzzy_match(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["John", "John", "Person"]])
    result = resolver.resolve([("Jon", "Person")], "Jon went home")
    assert result[0][1] == "John"


def test_fuzzy_below_threshold(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["John", "John", "Person"]])
    result = resolver.resolve([("Jonathan", "Person")], "Jonathan went home")
    assert result[0][1] == "Jonathan"


def test_new_entity(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [])
    result = resolver.resolve([("NewPerson", "Person")], "NewPerson joined")
    assert result == [("NewPerson", "NewPerson", "Person")]
    assert "newperson" in resolver._cache
    assert resolver._cache["newperson"] == "NewPerson"


def test_disabled(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entity_resolver, "ENTITY_RESOLVER_ENABLED", False)
    result = resolver.resolve(
        [("Alice", "Person"), ("Bob", "Person")],
        "Alice and Bob",
    )
    assert result == [("Alice", "Alice", "Person"), ("Bob", "Bob", "Person")]


def test_cache_invalidation(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(resolver, monkeypatch, [["Alice", "Alice", "Person"]])
    assert resolver._cache
    assert resolver._loaded_graph == "test_brain"

    resolver.invalidate_cache()

    assert not resolver._cache
    assert resolver._loaded_graph == ""


def test_multiple_entities(resolver: EntityResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(
        resolver,
        monkeypatch,
        [
            ["Alice", "Alice", "Person"],
            ["Google", "Google", "Organization"],
        ],
    )
    result = resolver.resolve(
        [("Alice", "Person"), ("google", "Organization"), ("NewEntity", "Person")],
        "Alice works at google and knows NewEntity",
    )
    assert result[0] == ("Alice", "Alice", "Person")
    assert result[1] == ("google", "Google", "Organization")
    assert result[2] == ("NewEntity", "NewEntity", "Person")
