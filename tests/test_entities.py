import importlib


hybrid_brain = importlib.import_module("hybrid_brain")


def test_extract_known_person(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "_load_known_entities", lambda: ({"John Doe"}, set(), set()))
    entities = hybrid_brain.extract_entities_fast("Yesterday John Doe reviewed the roadmap")
    assert ("John Doe", "Person") in entities


def test_extract_known_org(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "_load_known_entities", lambda: (set(), {"OpenClaw"}, set()))
    entities = hybrid_brain.extract_entities_fast("We integrated with OpenClaw this week")
    assert ("OpenClaw", "Organization") in entities


def test_no_substring_match(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "_load_known_entities", lambda: ({"Al"}, set(), set()))
    entities = hybrid_brain.extract_entities_fast("We improved algorithm performance")
    assert ("Al", "Person") not in entities


def test_capitalized_phrase_extraction(monkeypatch):
    monkeypatch.setattr(hybrid_brain, "_load_known_entities", lambda: (set(), set(), set()))
    entities = hybrid_brain.extract_entities_fast("Roadmap review with Jane Smith in Toronto")
    assert ("Jane Smith", "Person") in entities


def test_entity_extraction_consistency(monkeypatch):
    monkeypatch.setattr(
        hybrid_brain,
        "_load_known_entities",
        lambda: ({"John Doe"}, {"OpenClaw"}, {"Rasputin Memory"}),
    )
    text = "John Doe discussed Rasputin Memory rollout at OpenClaw"
    first = hybrid_brain.extract_entities_fast(text)
    second = hybrid_brain.extract_entities_fast(text)
    assert first == second
