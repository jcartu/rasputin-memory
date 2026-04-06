from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

FACT_EXTRACTION_MODEL = os.environ.get("FACT_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
FACT_EXTRACTION_PROVIDER = os.environ.get("FACT_EXTRACTION_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EXTRACTION_PROMPT = """Extract discrete facts from this conversation text. Each fact should be self-contained — understandable without the original context.

For each fact, provide:
- what: What happened or was stated (be specific — use exact names, numbers, items)
- who: People involved (resolve pronouns: "my roommate Emily" → "Emily (user's roommate)")
- when: When it happened (resolve relative dates using the Event Date)
- entities: Named entities mentioned

Skip greetings, filler, and trivial small talk. Keep decisions, preferences, relationships, events, facts about people, plans, emotions.

CRITICAL: If text says "yesterday" and Event Date is "2023-05-08", the fact happened on 2023-05-07. Always resolve relative dates.

Event Date: {event_date}

Text:
{text}

Return a JSON array. Example:
[{{"what": "Emily got promoted to senior engineer at Google", "who": "Emily", "when": "June 2024", "entities": ["Emily", "Google"]}}]

Return ONLY the JSON array."""


def extract_facts(
    text: str,
    event_date: str = "",
    source: str = "",
) -> list[dict[str, Any]]:
    if not text or len(text.strip()) < 30:
        return []

    if not event_date:
        from datetime import datetime, timezone

        event_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = EXTRACTION_PROMPT.format(event_date=event_date, text=text[:4000])

    try:
        import urllib.request

        if FACT_EXTRACTION_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            body = json.dumps(
                {
                    "model": FACT_EXTRACTION_MODEL,
                    "max_tokens": 2000,
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            content = data["content"][0]["text"].strip()
        else:
            import requests as _req

            resp = _req.post(
                os.environ.get("FACT_EXTRACTION_URL", "http://localhost:11434/v1/chat/completions"),
                json={
                    "model": FACT_EXTRACTION_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 2000,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip thinking tags if present (qwen3.5 sometimes wraps in <think>)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        facts = _parse_facts(content)
        for f in facts:
            f["source"] = source
            f["event_date"] = event_date
        return facts

    except Exception as e:
        logger.warning("Fact extraction failed: %s", e)
        return [{"what": text[:500], "who": "", "when": "", "entities": [], "source": source, "event_date": event_date}]


def _parse_facts(content: str) -> list[dict[str, Any]]:
    if "```" in content:
        parts = content.split("```")
        content = parts[1] if len(parts) >= 3 else parts[-1]
        if content.startswith("json"):
            content = content[4:]

    start = content.find("[")
    end = content.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            facts = json.loads(content[start:end])
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, dict) and f.get("what")]
        except json.JSONDecodeError:
            pass
    return []


def fact_to_text(fact: dict[str, Any]) -> str:
    parts = []
    what = fact.get("what", "")
    if what:
        parts.append(what)
    who = fact.get("who", "")
    if who and who not in ("N/A", ""):
        parts.append(f"Involving: {who}")
    when = fact.get("when", "")
    if when and when not in ("N/A", ""):
        parts.append(f"When: {when}")
    return " | ".join(parts)
