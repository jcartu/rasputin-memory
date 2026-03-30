"""Shared test fixtures for rasputin-memory tests."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@dataclass
class MockPoint:
    id: Any
    score: float = 0.0
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None


class MockQdrant:
    def __init__(self):
        self.points: dict[Any, dict[str, Any]] = {}

    def query_points(self, query=None, limit=10, with_payload=True, with_vectors=False, **_kwargs):
        query = query or []

        def cosine(a, b):
            if not a or not b:
                return 0.0
            n = min(len(a), len(b))
            a = a[:n]
            b = b[:n]
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        ranked = []
        for pid, data in self.points.items():
            score = cosine(query, data.get("vector", [])) if query else 0.0
            ranked.append(
                MockPoint(
                    id=pid,
                    score=score,
                    payload=data.get("payload") if with_payload else None,
                    vector=data.get("vector") if with_vectors else None,
                )
            )

        ranked.sort(key=lambda p: p.score, reverse=True)
        return SimpleNamespace(points=ranked[:limit])

    def upsert(self, collection_name=None, points=None, **_kwargs):
        for p in points or []:
            self.points[p.id] = {
                "vector": list(p.vector) if p.vector is not None else [],
                "payload": dict(p.payload) if p.payload is not None else {},
                "collection": collection_name,
            }

    def scroll(self, collection_name=None, limit=100, offset=None, with_payload=True, with_vectors=False, **_kwargs):
        ids = sorted(self.points.keys())
        if offset is None:
            start = 0
        else:
            start = ids.index(offset) + 1 if offset in ids else 0
        page_ids = ids[start : start + limit]
        next_offset = page_ids[-1] if (start + limit) < len(ids) else None
        points = [
            MockPoint(
                id=pid,
                payload=self.points[pid].get("payload") if with_payload else None,
                vector=self.points[pid].get("vector") if with_vectors else None,
            )
            for pid in page_ids
        ]
        return points, next_offset

    def retrieve(self, collection_name=None, ids=None, with_payload=True, with_vectors=False, **_kwargs):
        out = []
        for pid in ids or []:
            if pid not in self.points:
                continue
            out.append(
                MockPoint(
                    id=pid,
                    payload=self.points[pid].get("payload") if with_payload else None,
                    vector=self.points[pid].get("vector") if with_vectors else None,
                )
            )
        return out

    def set_payload(self, collection_name=None, points=None, payload=None, **_kwargs):
        for pid in points or []:
            if pid in self.points:
                existing = self.points[pid].setdefault("payload", {})
                existing.update(payload or {})

    def get_collection(self, collection_name=None):
        return SimpleNamespace(
            points_count=len(self.points), config=SimpleNamespace(params=SimpleNamespace(vectors={}))
        )


class MockRedis:
    def __init__(self):
        self.calls: list[tuple[Any, ...]] = []

    def ping(self):
        return True

    def execute_command(self, *args):
        self.calls.append(args)
        if len(args) >= 3 and args[0] == "GRAPH.QUERY":
            query = str(args[2])
            if "count(n)" in query:
                return [[], [[0]]]
            if "count(e)" in query:
                return [[], [[0]]]
            return [[], []]
        return "OK"


@pytest.fixture
def mock_qdrant() -> MockQdrant:
    return MockQdrant()


@pytest.fixture
def mock_embedding():
    def fake_embed(text: str, prefix: str = "") -> list[float]:
        seed_src = f"{prefix}::{text}".encode("utf-8")
        seed = int(hashlib.sha256(seed_src).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.random() for _ in range(768)]

    return fake_embed


@pytest.fixture
def mock_redis() -> MockRedis:
    return MockRedis()
