"""
OC Provider Adapter for RASPUTIN Memory System
Routes LLM/embedding/reranker calls through cloud APIs with Ollama fallback.

Cloud providers:
  - Embeddings: OpenAI text-embedding-3-small (1536d)
  - LLM/A-MAC:  OpenAI gpt-4o-mini
  - Reranking:   Cohere Rerank v3.5

Fallback (no API keys):
  - Embeddings: Ollama nomic-embed-text (768d)
  - LLM/A-MAC:  Ollama (local model)
  - Reranking:   Local reranker at localhost:8006

Part of OC-203 Phase 2 — Cloud Adaptation.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("rasputin.provider")

# Config paths
OC_CONFIG_PATH = os.environ.get(
    "OPENCLAW_CONFIG", os.path.expanduser("~/.openclaw/openclaw.json")
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LOCAL_RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8006/rerank")


class OCProvider:
    """
    Routes RASPUTIN model calls to cloud APIs or local Ollama fallback.
    Auto-detects available credentials at init time.
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = None
        self._openai_key = ""
        self._cohere_key = ""
        self._load_config(config_path or OC_CONFIG_PATH)
        self._detect_providers()

    # ── Config loading ─────────────────────────────────────────────────────

    def _load_config(self, path: str):
        """Load OC config for provider hints. Never fails."""
        try:
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    self._config = json.load(f)
        except Exception:
            pass  # Graceful — fall back to env vars

    def _detect_providers(self):
        """Detect available API keys from env vars (primary) or OC config."""
        # OpenAI
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not self._openai_key and self._config:
            # Try to extract from OC config secrets
            try:
                secrets = self._config.get("secrets", {})
                providers = secrets.get("providers", {})
                for prov in providers.values():
                    if isinstance(prov, dict) and prov.get("source") == "env":
                        # Check env for common OpenAI key names
                        for key_name in ("OPENAI_API_KEY", "OC_OPENAI_API_KEY"):
                            val = os.environ.get(key_name, "")
                            if val:
                                self._openai_key = val
                                break
            except Exception:
                pass

        # Cohere
        self._cohere_key = os.environ.get("COHERE_API_KEY", "")

        # Log what we found
        if self._openai_key:
            log.info("Provider: OpenAI (text-embedding-3-small + gpt-4o-mini)")
        else:
            log.warning("No OPENAI_API_KEY — falling back to Ollama for embeddings + LLM")

        if self._cohere_key:
            log.info("Provider: Cohere (rerank-v3.5)")
        else:
            log.warning("No COHERE_API_KEY — falling back to local reranker at :8006")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def embedding_provider(self) -> str:
        return "openai" if self._openai_key else "ollama"

    @property
    def amac_provider(self) -> str:
        return "openai" if self._openai_key else "ollama"

    @property
    def reranker_provider(self) -> str:
        return "cohere" if self._cohere_key else "local"

    # ── Embeddings ─────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector.
        Cloud: OpenAI text-embedding-3-small → 1536d
        Fallback: Ollama nomic-embed-text → 768d
        """
        if self._openai_key:
            return self._openai_embedding(text)
        return self._ollama_embedding(text)

    def get_embedding_dim(self) -> int:
        """Returns embedding dimensionality for current provider."""
        return 1536 if self._openai_key else 768

    def _openai_embedding(self, text: str) -> list[float]:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._openai_key}"},
            json={"model": "text-embedding-3-small", "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def _ollama_embedding(self, text: str) -> list[float]:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"embeddings": [[...]]} for /api/embed
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        # Older Ollama format
        return data.get("embedding", [])

    # ── LLM Completion ─────────────────────────────────────────────────────

    def llm_completion(self, messages: list[dict], max_tokens: int = 500,
                       temperature: float = 0.05) -> str:
        """
        Generate LLM completion.
        Cloud: gpt-4o-mini via OpenAI API
        Fallback: Ollama local model
        """
        if self._openai_key:
            return self._openai_completion(messages, max_tokens, temperature)
        return self._ollama_completion(messages, max_tokens, temperature)

    def _openai_completion(self, messages: list[dict], max_tokens: int,
                           temperature: float) -> str:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._openai_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _ollama_completion(self, messages: list[dict], max_tokens: int,
                           temperature: float) -> str:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "qwen2.5:7b",
                "messages": messages,
                "options": {"num_predict": max_tokens, "temperature": temperature},
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ── A-MAC Scoring ──────────────────────────────────────────────────────

    def amac_score(self, text: str) -> Optional[tuple[float, float, float, float]]:
        """
        Score text on Relevance, Novelty, Specificity via LLM.
        Returns (relevance, novelty, specificity, composite) or None on failure.
        Uses the LAST triplet found (thinking models may echo examples).
        """
        prompt = (
            'You are a memory quality filter. Score the following memory on 3 dimensions.\n'
            'Return ONLY three integers separated by commas. No text, no explanation. '
            'Just three numbers like: 7,4,8\n\n'
            'Relevance 0-10: Is this about your user\'s key domains? '
            '(0=totally unrelated, 10=highly relevant)\n'
            'Novelty 0-10: Does this add genuinely NEW, specific information? '
            '(0=generic platitude, 10=unique concrete fact)\n'
            'Specificity 0-10: Is this a concrete verifiable fact with numbers/names/dates? '
            '(0=vague filler, 10=specific actionable data)\n\n'
            'Examples:\n'
            '"Things are going well." -> 0,1,0\n'
            '"The weather is nice today." -> 1,1,1\n'
            '"Project Alpha deadline moved to Q3 2026, budget approved at $500K." -> 9,8,10\n'
            '"Alice joined the backend team, previously at Acme Corp for 3 years." -> 8,9,9\n\n'
            f'Memory: "{text[:800]}"\n\n'
            'Output format: R,N,S (three integers separated by commas, nothing else)'
        )

        try:
            response = self.llm_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.05,
            )

            # Find ALL R,N,S triplets — use the LAST one (thinking models echo examples)
            triplets = re.findall(r'(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})', response)
            if not triplets:
                log.warning(f"A-MAC: no triplet found in response: {response[:100]}")
                return None

            last = triplets[-1]
            r, n, s = float(last[0]), float(last[1]), float(last[2])
            composite = (r + n + s) / 3.0
            return (r, n, s, composite)

        except Exception as e:
            log.warning(f"A-MAC scoring failed: {e}")
            return None

    # ── Reranking ──────────────────────────────────────────────────────────

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """
        Rerank passages by relevance to query.
        Cloud: Cohere Rerank v3.5
        Fallback: Local reranker at localhost:8006
        Returns list of scores (one per passage, in original order).
        """
        if not passages:
            return []

        if self._cohere_key:
            return self._cohere_rerank(query, passages)
        return self._local_rerank(query, passages)

    def _cohere_rerank(self, query: str, passages: list[str]) -> list[float]:
        import cohere
        co = cohere.ClientV2(api_key=self._cohere_key)
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=passages,
            top_n=len(passages),
        )
        # Build scores array in original passage order
        scores = [0.0] * len(passages)
        for r in response.results:
            scores[r.index] = r.relevance_score
        return scores

    def _local_rerank(self, query: str, passages: list[str]) -> list[float]:
        try:
            resp = requests.post(
                LOCAL_RERANKER_URL,
                json={"query": query, "passages": passages},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Expect {"scores": [...]} from local reranker
            return data.get("scores", [0.0] * len(passages))
        except Exception as e:
            log.warning(f"Local reranker unavailable: {e}")
            # Return zeros — caller will fall back to vector order
            return [0.0] * len(passages)

    # ── Health / Info ──────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return provider configuration summary."""
        return {
            "embedding": self.embedding_provider,
            "embedding_dim": self.get_embedding_dim(),
            "embedding_model": "text-embedding-3-small" if self._openai_key else "nomic-embed-text",
            "llm": self.amac_provider,
            "llm_model": "gpt-4o-mini" if self._openai_key else "ollama/qwen2.5:7b",
            "reranker": self.reranker_provider,
            "reranker_model": "rerank-v3.5" if self._cohere_key else "local:8006",
        }

    def health_check(self) -> dict:
        """Check connectivity to all providers."""
        status = {}

        # Embeddings
        try:
            vec = self.get_embedding("test")
            status["embeddings"] = f"ok ({len(vec)}d via {self.embedding_provider})"
        except Exception as e:
            status["embeddings"] = f"error: {e}"

        # LLM
        try:
            result = self.llm_completion(
                [{"role": "user", "content": "Reply with OK"}],
                max_tokens=5,
            )
            status["llm"] = f"ok ({self.amac_provider}: {result.strip()[:20]})"
        except Exception as e:
            status["llm"] = f"error: {e}"

        # Reranker
        try:
            scores = self.rerank("test", ["hello world", "unrelated topic"])
            status["reranker"] = f"ok ({self.reranker_provider}: {len(scores)} scores)"
        except Exception as e:
            status["reranker"] = f"error: {e}"

        return status


# ── Singleton ──────────────────────────────────────────────────────────────

_provider: Optional[OCProvider] = None


def get_provider() -> OCProvider:
    """Get or create the singleton OCProvider instance."""
    global _provider
    if _provider is None:
        _provider = OCProvider()
    return _provider


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print("OC Provider Adapter — Testing...")
    p = OCProvider()
    print(f"\nProvider info: {json.dumps(p.info(), indent=2)}")
    print("\nHealth check:")
    status = p.health_check()
    for component, result in status.items():
        print(f"  {component}: {result}")
