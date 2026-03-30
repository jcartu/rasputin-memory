import importlib


entities = importlib.import_module("brain.entities")


def test_extract_known_person(monkeypatch):
    monkeypatch.setattr(entities, "_load_known_entities", lambda: ({"John Doe"}, set(), set()))
    extracted = entities.extract_entities_fast("Yesterday John Doe reviewed the roadmap")
    assert ("John Doe", "Person") in extracted


def test_extract_known_org(monkeypatch):
    monkeypatch.setattr(entities, "_load_known_entities", lambda: (set(), {"OpenClaw"}, set()))
    extracted = entities.extract_entities_fast("We integrated with OpenClaw this week")
    assert ("OpenClaw", "Organization") in extracted


def test_no_substring_match(monkeypatch):
    monkeypatch.setattr(entities, "_load_known_entities", lambda: ({"Al"}, set(), set()))
    extracted = entities.extract_entities_fast("We improved algorithm performance")
    assert ("Al", "Person") not in extracted


def test_capitalized_phrase_extraction(monkeypatch):
    monkeypatch.setattr(entities, "_load_known_entities", lambda: (set(), set(), set()))
    extracted = entities.extract_entities_fast("Roadmap review with Jane Smith in Toronto")
    assert ("Jane Smith", "Person") in extracted


def test_entity_extraction_consistency(monkeypatch):
    monkeypatch.setattr(
        entities,
        "_load_known_entities",
        lambda: ({"John Doe"}, {"OpenClaw"}, {"Rasputin Memory"}),
    )
    text = "John Doe discussed Rasputin Memory rollout at OpenClaw"
    first = entities.extract_entities_fast(text)
    second = entities.extract_entities_fast(text)
    assert first == second
