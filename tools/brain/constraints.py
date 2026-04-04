from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

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
CONSTRAINTS_PROVIDER = os.environ.get(
    "CONSTRAINTS_PROVIDER",
    _state.CONFIG.get("constraints", {}).get("provider", "anthropic"),
)
CONSTRAINTS_MODEL = os.environ.get(
    "CONSTRAINTS_MODEL",
    _state.CONFIG.get("constraints", {}).get("model", "claude-sonnet-4-6"),
)
CONSTRAINTS_TIMEOUT = int(_state.CONFIG.get("constraints", {}).get("timeout", 15))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def extract_constraints(text: str) -> list[dict[str, Any]]:
    if not text or len(text) < 30:
        return []

    prompt = _EXTRACTION_PROMPT.format(text=text[:2000])

    if CONSTRAINTS_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        return _extract_anthropic(prompt)
    else:
        return _extract_local(prompt)


def _extract_anthropic(prompt: str) -> list[dict[str, Any]]:
    body = json.dumps(
        {
            "model": CONSTRAINTS_MODEL,
            "max_tokens": 500,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", ANTHROPIC_API_KEY)
    req.add_header("anthropic-version", "2023-06-01")

    try:
        with urllib.request.urlopen(req, timeout=CONSTRAINTS_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        raw = data["content"][0]["text"].strip()
        return _parse_constraints(raw)
    except Exception as exc:
        _state.logger.debug("Anthropic constraint extraction failed: %s", exc)
        return []


def _extract_local(prompt: str) -> list[dict[str, Any]]:
    import requests

    try:
        resp = requests.post(
            _state.AMAC_LLM_URL,
            json={
                "model": _state.CONFIG.get("constraints", {}).get("model", "qwen3.5:9b"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=CONSTRAINTS_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return _parse_constraints(raw)
    except Exception as exc:
        _state.logger.debug("Local constraint extraction failed: %s", exc)
        return []


def _parse_constraints(raw: str) -> list[dict[str, Any]]:
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
