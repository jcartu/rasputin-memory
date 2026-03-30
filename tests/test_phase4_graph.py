import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import hybrid_brain


def test_ner_consistency(monkeypatch):
    monkeypatch.setattr(
        hybrid_brain,
        "_load_known_entities",
        lambda: ({"John Doe"}, {"BrandA", "OpenClaw"}, {"Rasputin Memory"}),
    )

    text = "John Doe built Rasputin Memory with OpenClaw at BrandA."

    write_entities = hybrid_brain.extract_entities_fast(text)
    read_entities = hybrid_brain.extract_entities_fast(text)

    assert set(write_entities) == set(read_entities)
    assert ("John Doe", "Person") in write_entities
    assert ("OpenClaw", "Organization") in write_entities
