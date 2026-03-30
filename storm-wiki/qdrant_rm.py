#!/usr/bin/env python3
"""
Qdrant VectorRM adapter for STORM Wiki Generator.
Embeds via Ollama nomic-embed-text (port 11434), queries Qdrant (port 6333).
Uses the existing 'second_brain' collection.
"""

import requests
from typing import List, Union, Optional

import dspy


class QdrantLocalRM(dspy.Retrieve):
    """Retrieve from local Qdrant using Ollama embeddings."""

    def __init__(
        self,
        collection_name: str = "second_brain",
        qdrant_url: str = "http://localhost:6333",
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        k: int = 5,
    ):
        super().__init__(k=k)
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url.rstrip("/")
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_model = embed_model
        self.usage = 0

    def _embed(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.embed_model, "input": f"search_query: {text}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"embeddings": [[...]]} for /api/embed
        return data["embeddings"][0]

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"QdrantLocalRM": usage}

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        exclude_urls: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search Qdrant for top-k passages.

        Returns list of dicts with keys: description, snippets, title, url
        (matching STORM's expected format).
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected = []

        for query in queries:
            vector = self._embed(query)

            resp = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json={
                    "vector": vector,
                    "limit": self.k,
                    "with_payload": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])

            for hit in results:
                payload = hit.get("payload", {})
                text = payload.get("text", payload.get("content", ""))
                source = payload.get("source", "memory")
                ts = payload.get("timestamp", "")
                doc_id = hit.get("id", "unknown")

                collected.append({
                    "description": f"Memory from {source} ({ts})",
                    "snippets": [text],
                    "title": f"{source}: {text[:80]}...",
                    "url": f"qdrant://{self.collection_name}/{doc_id}",
                })

        return collected


if __name__ == "__main__":
    rm = QdrantLocalRM()
    results = rm.forward("user's family")
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['title']}")
        print(f"    {r['snippets'][0][:200]}")
