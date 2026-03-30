"""
hybrid_brain — Rasputin Memory hybrid search library.

Provides clean, importable classes for vector + BM25 + graph + reranker
search over a Qdrant memory store.

Quick start::

    from hybrid_brain import HybridSearch

    search = HybridSearch(collection="second_brain")
    results = search.query("Sasha IVF meeting", limit=5)
    for r in results:
        print(f"{r['score']:.3f}  {r['text'][:80]}")

Individual components::

    from hybrid_brain import RRFFusion, TemporalDecay, QualityGate

    fuser = RRFFusion(k=60)
    merged = fuser.fuse([vec_ids, bm25_ids])

    decay = TemporalDecay()
    results = decay.apply(results)

    gate = QualityGate(threshold=4.0)
    if gate.evaluate(text).admitted:
        ...

Classes
-------
- :class:`~hybrid_brain.search.HybridSearch` — main search orchestrator
- :class:`~hybrid_brain.fusion.RRFFusion` — Reciprocal Rank Fusion
- :class:`~hybrid_brain.temporal.TemporalDecay` — Ebbinghaus temporal decay
- :class:`~hybrid_brain.quality_gate.QualityGate` — AMAC admission control
"""

from .fusion import RRFFusion
from .quality_gate import QualityGate
from .search import HybridSearch
from .temporal import TemporalDecay

__all__ = ["HybridSearch", "RRFFusion", "TemporalDecay", "QualityGate"]
__version__ = "0.3.0"
