from __future__ import annotations

import re
from typing import Any


NEGATION_PATTERNS = (
    r"\bnot\b",
    r"\bno longer\b",
    r"\bstopped\b",
    r"\bnever\b",
    r"\bdoesn't\b",
    r"\bdidn't\b",
    r"\bisn't\b",
    r"\baren't\b",
)

_WORD_RE = re.compile(r"\w+")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _contains_negation(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in NEGATION_PATTERNS)


def _extract_subject_number_pairs(text: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    lowered = text.lower()
    for m in re.finditer(
        r"([a-z][a-z0-9\s]{2,40}?)\s(?:is|was|at|has|have|had|costs?|worth|salary|income|age)\s+\$?([0-9]+(?:\.[0-9]+)?)",
        lowered,
    ):
        subject = " ".join(m.group(1).split())
        value = float(m.group(2))
        if subject:
            pairs.append((subject, value))
    return pairs


def _extract_subject_location_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    normalized = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"([A-Z][A-Za-z0-9\s]{1,40}?)\s(?:moved to|relocated to|lives in|is in|is at|located in)\s([A-Z][A-Za-z\s\.-]{1,40})",
        r"([A-Z][A-Za-z0-9\s]{1,40}?)\s(?:from)\s([A-Z][A-Za-z\s\.-]{1,40})",
    ]

    for pat in patterns:
        for m in re.finditer(pat, normalized):
            subject = " ".join(m.group(1).split()).lower()
            location = " ".join(m.group(2).split()).lower()
            if subject and location:
                pairs.append((subject, location))
    return pairs


def _shared_subject_context(new_text: str, old_text: str) -> bool:
    new_words = set(_WORD_RE.findall(new_text.lower()))
    old_words = set(_WORD_RE.findall(old_text.lower()))
    overlap = new_words & old_words
    if len(overlap) >= 3:
        return True

    new_names = {m.group(1).lower() for m in _NAME_RE.finditer(new_text)}
    old_names = {m.group(1).lower() for m in _NAME_RE.finditer(old_text)}
    return bool(new_names & old_names)


def looks_contradictory(new_text: str, old_text: str) -> bool:
    if not new_text or not old_text:
        return False

    shared_subject = _shared_subject_context(new_text, old_text)

    if shared_subject and _contains_negation(new_text) != _contains_negation(old_text):
        return True

    new_numbers = _extract_subject_number_pairs(new_text)
    old_numbers = _extract_subject_number_pairs(old_text)
    for new_subject, new_value in new_numbers:
        for old_subject, old_value in old_numbers:
            if new_subject == old_subject and new_value != old_value:
                return True

    new_locations = _extract_subject_location_pairs(new_text)
    old_locations = _extract_subject_location_pairs(old_text)
    for new_subject, new_location in new_locations:
        for old_subject, old_location in old_locations:
            if new_subject == old_subject and new_location != old_location:
                return True

    return False


def llm_verify_contradiction(new_text: str, old_text: str) -> bool:
    import os

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return False

    prompt = (
        "Do these two statements contradict each other? "
        "A statement CONTRADICTS another if accepting one requires rejecting the other.\n\n"
        f'Statement A: "{new_text[:500]}"\n'
        f'Statement B: "{old_text[:500]}"\n\n'
        'Reply with ONLY "YES" or "NO".'
    )

    try:
        import json
        import urllib.request

        body = json.dumps(
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 10,
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
        req.add_header("x-api-key", anthropic_key)
        req.add_header("anthropic-version", "2023-06-01")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return "YES" in data["content"][0]["text"].strip().upper()
    except Exception:
        return False


def check_contradictions(
    text: str,
    embedding: list[float],
    qdrant_client: Any,
    collection: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    try:
        similar = qdrant_client.query_points(
            collection_name=collection,
            query=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=0.85,
        )
    except Exception:
        return []

    points = getattr(similar, "points", similar)
    contradictions: list[dict[str, Any]] = []

    for point in points or []:
        payload = getattr(point, "payload", None) or {}
        existing_text = payload.get("text", "")
        score = getattr(point, "score", 0.0) or 0.0
        if score < 0.85 or not existing_text:
            continue
        if looks_contradictory(text, existing_text):
            contradictions.append(
                {
                    "existing_id": getattr(point, "id", None),
                    "existing_text": existing_text[:200],
                    "similarity": round(float(score), 4),
                }
            )
        elif score >= 0.90 and llm_verify_contradiction(text, existing_text):
            contradictions.append(
                {
                    "existing_id": getattr(point, "id", None),
                    "existing_text": existing_text[:200],
                    "similarity": round(float(score), 4),
                    "llm_verified": True,
                }
            )
    return contradictions
