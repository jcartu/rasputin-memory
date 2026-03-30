"""
hybrid_brain.quality_gate — AMAC (Adaptive Memory Admission Control) quality gate.

Scores incoming text with an LLM and rejects low-quality memories before they
pollute the vector store.

Example::

    from hybrid_brain import QualityGate

    gate = QualityGate(threshold=4.0)
    result = gate.evaluate("The sky is blue.")
    if result.admitted:
        print(f"Score: {result.score}/10 — admitted")
    else:
        print(f"Score: {result.score}/10 — rejected: {result.reason}")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

AMAC_TIMEOUT = 30
AMAC_DEFAULT_MODEL = "qwen3.5:35b"
AMAC_REJECT_LOG = "/tmp/amac_rejected.log"
OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class GateResult:
    """Result of :meth:`QualityGate.evaluate`."""

    admitted: bool
    """Whether the text passed the quality gate."""

    score: float
    """Quality score (0–10)."""

    reason: str
    """Short explanation (LLM-generated or rule-based)."""

    raw_response: Optional[str] = None
    """Raw LLM response for debugging."""


class QualityGate:
    """AMAC quality gate — scores text with an LLM before memory admission.

    Parameters
    ----------
    threshold:
        Minimum score (0–10) required for admission (default 4.0).
    model:
        Ollama model to use for scoring.
    enabled:
        Set ``False`` to bypass the gate entirely (admit everything).

    Example::

        gate = QualityGate(threshold=4.0)
        result = gate.evaluate("Josh had a meeting with Sarah about Q3 targets.")
        if result.admitted:
            commit_memory(text)
    """

    PROMPT_TEMPLATE = """Rate the long-term memory value of this text snippet on a scale of 0-10.

Scoring criteria:
- 8-10: Highly specific facts, decisions, preferences, relationships, or events with clear future utility
- 5-7: Somewhat useful context, general information, partial facts
- 2-4: Vague statements, generic observations, filler content
- 0-1: Gibberish, test data, spam, or content with no memory value

Text to evaluate:
\"\"\"{text}\"\"\"

Respond with ONLY a JSON object: {{"score": <number 0-10>, "reason": "<brief one-line explanation>"}}"""

    def __init__(
        self,
        threshold: float = 4.0,
        model: str = AMAC_DEFAULT_MODEL,
        enabled: bool = True,
    ) -> None:
        self.threshold = threshold
        self.model = model
        self.enabled = enabled

    def evaluate(self, text: str, source: str = "unknown", force: bool = False) -> GateResult:
        """Evaluate *text* for memory admission.

        Parameters
        ----------
        text:
            The text to evaluate.
        source:
            Source identifier (used in rejection logs).
        force:
            If ``True``, bypass threshold check (always admit if LLM scores it).

        Returns
        -------
        :class:`GateResult`
        """
        if not self.enabled:
            return GateResult(admitted=True, score=10.0, reason="gate disabled")

        try:
            score, reason, raw = self._llm_score(text)
        except Exception as e:
            logger.warning("AMAC LLM error (%s) — admitting by default", e)
            return GateResult(admitted=True, score=5.0, reason=f"llm_error: {e}")

        admitted = force or score >= self.threshold
        if not admitted:
            self._log_rejection(text, source, score, reason)

        return GateResult(admitted=admitted, score=score, reason=reason, raw_response=raw)

    def _llm_score(self, text: str) -> tuple[float, str, str]:
        """Call the Ollama LLM and parse score + reason."""
        prompt = self.PROMPT_TEMPLATE.format(text=text[:2000])
        resp = requests.post(
            OLLAMA_URL,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=AMAC_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        # Parse JSON from response
        match = re.search(r'\{.*?"score"\s*:\s*([0-9.]+).*?\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return float(data["score"]), data.get("reason", ""), raw
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Fallback: extract bare number
        num_match = re.search(r'\b([0-9](?:\.[0-9]+)?|10)\b', raw)
        if num_match:
            return float(num_match.group(1)), "parsed_from_text", raw

        raise ValueError(f"Could not parse AMAC score from: {raw[:200]}")

    def _log_rejection(self, text: str, source: str, score: float, reason: str) -> None:
        try:
            with open(AMAC_REJECT_LOG, "a") as f:
                f.write(
                    json.dumps({
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "source": source,
                        "score": score,
                        "reason": reason,
                        "text_preview": text[:200],
                    }) + "\n"
                )
        except OSError:
            pass
