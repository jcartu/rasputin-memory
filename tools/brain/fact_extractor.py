from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

FACT_EXTRACTION_MODEL = os.environ.get("FACT_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
FACT_EXTRACTION_PROVIDER = os.environ.get("FACT_EXTRACTION_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = os.environ.get("CEREBRAS_FACT_MODEL", "qwen-3-235b-a22b-instruct-2507")
_CEREBRAS_MAX_FAILURES = 5
_cerebras_consecutive_failures = 0


class ExtractedEntity(BaseModel):
    name: str = Field(description="Named entity as it appears")
    type: str = Field(description="Person, Organization, Location, Event, Thing")


class ExtractedFact(BaseModel):
    what: str = Field(description="Core fact — concise, self-contained (1-2 sentences)")
    who: Optional[str] = Field(default="N/A", description="People involved, pronouns resolved")
    when: Optional[str] = Field(default="N/A", description="When it happened, relative dates resolved")
    where: Optional[str] = Field(default="N/A", description="Location if relevant")
    fact_type: Literal["world", "experience", "inference"] = Field(
        default="world",
        description="world=objective, experience=subjective, inference=reasonable conclusion",
    )
    occurred_start: Optional[str] = Field(
        default=None,
        description="ISO date when this happened. Resolve relative: 'yesterday' + event_date 2023-05-08 = 2023-05-07",
    )
    occurred_end: Optional[str] = Field(
        default=None,
        description="ISO date when this ended. None for point-in-time",
    )
    entities: list[ExtractedEntity] = Field(default_factory=list)
    confidence: float = Field(
        default=0.8,
        description="0.9 for explicit statements, 0.7 for inferences, 0.5 for weak signals",
    )


class FactExtractionResponse(BaseModel):
    facts: list[ExtractedFact]


EXTRACTION_PROMPT = """Extract discrete facts AND reasonable inferences from this conversation.

RULES:
1. Resolve ALL pronouns to names using context: "she went" → "Caroline went"
2. Resolve relative dates using Event Date: "yesterday" + Event Date 2023-05-08 = 2023-05-07
3. Extract INFERENCES — reasonable conclusions from evidence:
   - Attends LGBTQ support groups → "likely supportive of LGBTQ causes" (inference, confidence 0.7)
   - Collects children's books → "would likely enjoy Dr. Seuss" (inference, confidence 0.6)
   - Had bad experience at restaurant → "probably wouldn't want to return there" (inference, confidence 0.7)
   Mark these as fact_type="inference" with confidence 0.6-0.7
4. fact_type: "world" for objective facts, "experience" for feelings/opinions, "inference" for conclusions
5. occurred_start/occurred_end: ISO dates when determinable
   - "worked at Google from 2021 to 2024" → occurred_start="2021-01-01", occurred_end="2024-12-31"
   - "yesterday" + Event Date 2023-05-08 → occurred_start="2023-05-07"

Event Date: {event_date}

Text:
{text}

Return ONLY a JSON object: {{"facts": [...]}}
Each fact: what, who, when, where, fact_type, occurred_start, occurred_end, entities, confidence

Example output:
{{"facts": [
  {{"what": "Caroline attended an LGBTQ support group meeting", "who": "Caroline", "when": "May 7, 2023", "where": "N/A", "fact_type": "world", "occurred_start": "2023-05-07", "occurred_end": null, "entities": [{{"name": "Caroline", "type": "Person"}}], "confidence": 0.9}},
  {{"what": "Caroline is likely supportive of LGBTQ causes based on her attendance at support group meetings", "who": "Caroline", "when": "N/A", "where": "N/A", "fact_type": "inference", "occurred_start": null, "occurred_end": null, "entities": [{{"name": "Caroline", "type": "Person"}}], "confidence": 0.7}}
]}}"""


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

        global _cerebras_consecutive_failures
        content = None

        if (
            CEREBRAS_API_KEY
            and FACT_EXTRACTION_PROVIDER != "anthropic_only"
            and _cerebras_consecutive_failures < _CEREBRAS_MAX_FAILURES
        ):
            try:
                body = json.dumps(
                    {
                        "model": CEREBRAS_MODEL,
                        "max_tokens": 2000,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ).encode()
                req = urllib.request.Request(CEREBRAS_URL, data=body, method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Authorization", f"Bearer {CEREBRAS_API_KEY}")
                req.add_header("User-Agent", "rasputin-memory/0.9.1")  # Cloudflare 1010 blocks default Python-urllib UA
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"].strip()
                _cerebras_consecutive_failures = 0
            except Exception as e:
                _cerebras_consecutive_failures += 1
                logger.warning(
                    "Cerebras extraction failed (%d/%d), falling back to Haiku: %s",
                    _cerebras_consecutive_failures,
                    _CEREBRAS_MAX_FAILURES,
                    e,
                )

        if (
            content is None
            and FACT_EXTRACTION_PROVIDER in ("anthropic", "anthropic_only", "cerebras")
            and ANTHROPIC_API_KEY
        ):
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

        if content is None:
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

        facts = _parse_extraction_response(content)
        for f in facts:
            f["source"] = source
            f["event_date"] = event_date
        return facts

    except Exception as e:
        logger.warning("Fact extraction failed: %s", e)
        return [{"what": text[:500], "who": "", "when": "", "entities": [], "source": source, "event_date": event_date}]


def _parse_extraction_response(response_text: str) -> list[dict[str, Any]]:
    """Parse LLM response with Pydantic validation and graceful fallback."""
    raw = response_text.strip()

    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 3 else parts[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    obj_start = raw.find("{")
    obj_end = raw.rfind("}") + 1
    if obj_start >= 0 and obj_end > obj_start:
        try:
            parsed = json.loads(raw[obj_start:obj_end])
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            try:
                response = FactExtractionResponse(**parsed)
                return [fact.model_dump() for fact in response.facts]
            except (ValidationError, TypeError) as e:
                logger.warning("Pydantic validation error, falling back to raw parse: %s", e)

            if "facts" in parsed and isinstance(parsed["facts"], list):
                return [f for f in parsed["facts"] if isinstance(f, dict) and f.get("what")]

    # Fallback: try bare JSON array (backward compat with old format)
    arr_start = raw.find("[")
    arr_end = raw.rfind("]") + 1
    if arr_start >= 0 and arr_end > arr_start:
        try:
            arr = json.loads(raw[arr_start:arr_end])
            if isinstance(arr, list):
                return [f for f in arr if isinstance(f, dict) and f.get("what")]
        except json.JSONDecodeError:
            pass

    logger.warning("No valid JSON found in extraction response")
    return []


def fact_to_text(fact: dict[str, Any]) -> str:
    parts: list[str] = []
    fact_type = fact.get("fact_type", "")
    if fact_type == "inference":
        parts.append("[Inference]")
    what = fact.get("what", "")
    if what:
        parts.append(what)
    who = fact.get("who", "")
    if who and who not in ("N/A", ""):
        parts.append(f"Involving: {who}")
    when = fact.get("when", "")
    if when and when not in ("N/A", ""):
        parts.append(f"When: {when}")
    where = fact.get("where", "")
    if where and where not in ("N/A", ""):
        parts.append(f"Where: {where}")
    start = fact.get("occurred_start")
    end = fact.get("occurred_end")
    if start:
        if end:
            parts.append(f"Period: {start} to {end}")
        else:
            parts.append(f"Date: {start}")
    return " | ".join(parts)
