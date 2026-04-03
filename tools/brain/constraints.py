"""Extract implicit constraints (goals, states, values, causal chains) from conversation text."""

from __future__ import annotations

import json
from typing import Any

import requests

from brain import _state

_EXTRACTION_PROMPT = """Analyze this conversation snippet and extract any IMPLICIT constraints that a good assistant should remember for future interactions.

Constraints are things like:
- GOAL: "The user wants to save money" (from "I've been cutting back on eating out")
- STATE: "The user is stressed about work" (from "This project deadline is killing me")
- VALUE: "The user values privacy" (from "I always use a VPN and encrypted messaging")
- CAUSAL: "The user avoids dairy because of lactose intolerance" (from "Last time I had cheese I felt terrible")

Text: "{text}"

If there are implicit constraints, return a JSON array:
[{{"type": "goal|state|value|causal", "constraint": "concise constraint description"}}]

If no implicit constraints exist, return: []
Return ONLY the JSON array."""

CONSTRAINTS_ENABLED = _state.CONFIG.get("constraints", {}).get("enabled", False)
CONSTRAINTS_MODEL = _state.CONFIG.get("constraints", {}).get("model", "qwen3.5:9b")
CONSTRAINTS_TIMEOUT = int(_state.CONFIG.get("constraints", {}).get("timeout", 15))


def extract_constraints(text: str) -> list[dict[str, Any]]:
    if not text or len(text) < 30:
        return []

    prompt = _EXTRACTION_PROMPT.format(text=text[:2000])
    try:
        resp = requests.post(
            _state.AMAC_LLM_URL,
            json={
                "model": CONSTRAINTS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=CONSTRAINTS_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Strip markdown fences
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) >= 3 else parts[-1]
            if raw.startswith("json"):
                raw = raw[4:]

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            constraints = json.loads(raw[start:end])
            if isinstance(constraints, list):
                return [c for c in constraints if isinstance(c, dict) and "constraint" in c][:10]
        return []
    except Exception as exc:
        _state.logger.debug("Constraint extraction failed: %s", exc)
        return []
