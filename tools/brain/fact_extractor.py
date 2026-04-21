from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib import error, request

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

FACT_EXTRACTION_MODEL = os.environ.get("FACT_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
FACT_EXTRACTION_PROVIDER = os.environ.get("FACT_EXTRACTION_PROVIDER", "local_vllm")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = os.environ.get("CEREBRAS_FACT_MODEL", "qwen-3-235b-a22b-instruct-2507")
LOCAL_VLLM_URL = os.environ.get("LOCAL_VLLM_URL", "http://localhost:11437/v1/chat/completions")
LOCAL_VLLM_MODEL = os.environ.get("LOCAL_VLLM_MODEL", "qwen3-32b-awq")
LOCAL_VLLM_SEED = int(os.environ.get("LOCAL_VLLM_SEED", "42"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_CEREBRAS_MAX_FAILURES = 5
_cerebras_consecutive_failures = 0
_EXTRACTION_PROVIDER_LOG = "/tmp/bench_runs/extraction_provider_log.jsonl"
_PROVIDER_ORDER = ("local_vllm", "cerebras", "groq", "anthropic")
_CURRENT_EXTRACTION_SESSION_ID = ""


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
        event_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = EXTRACTION_PROMPT.format(event_date=event_date, text=text[:4000])

    global _CURRENT_EXTRACTION_SESSION_ID
    _CURRENT_EXTRACTION_SESSION_ID = source
    try:
        facts = _extract_with_chain(prompt, event_date, FACT_EXTRACTION_PROVIDER)
    finally:
        _CURRENT_EXTRACTION_SESSION_ID = ""

    for fact in facts:
        fact["source"] = source
        fact["event_date"] = event_date
    return facts


def _call_local_vllm(prompt: str, event_date: str) -> list[dict[str, Any]] | None:
    del event_date
    # chat_template_kwargs.enable_thinking=false disables Qwen3 reasoning-mode <think>...</think>
    # blocks that would otherwise prefix the JSON output and break fact parsing.
    body = {
        "model": LOCAL_VLLM_MODEL,
        "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": LOCAL_VLLM_SEED,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return _call_provider(
        provider="local_vllm",
        prompt=prompt,
        url=LOCAL_VLLM_URL,
        body=body,
        timeout=60,
        headers={"Content-Type": "application/json", "User-Agent": "rasputin-memory/0.9.1"},
        content_parser=_extract_openai_content,
        token_parser=_extract_openai_tokens,
    )


def _call_cerebras(prompt: str, event_date: str) -> list[dict[str, Any]] | None:
    del event_date
    global _cerebras_consecutive_failures

    if not CEREBRAS_API_KEY:
        _log_provider_attempt(
            provider="cerebras",
            latency_ms=0,
            fact_count=0,
            fallback_reason="missing_api_key",
            input_token_count=_estimate_token_count(prompt),
            output_token_count=0,
        )
        return None

    if _cerebras_consecutive_failures >= _CEREBRAS_MAX_FAILURES:
        _log_provider_attempt(
            provider="cerebras",
            latency_ms=0,
            fact_count=0,
            fallback_reason="cerebras_circuit_open",
            input_token_count=_estimate_token_count(prompt),
            output_token_count=0,
        )
        return None

    body = {
        "model": CEREBRAS_MODEL,
        "max_tokens": 2000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    facts = _call_provider(
        provider="cerebras",
        prompt=prompt,
        url=CEREBRAS_URL,
        body=body,
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "User-Agent": "rasputin-memory/0.9.1",
        },
        content_parser=_extract_openai_content,
        token_parser=_extract_openai_tokens,
    )
    if facts is None:
        _cerebras_consecutive_failures += 1
        logger.warning(
            "Cerebras extraction failed (%d/%d), falling through",
            _cerebras_consecutive_failures,
            _CEREBRAS_MAX_FAILURES,
        )
        return None

    _cerebras_consecutive_failures = 0
    return facts


def _call_groq(prompt: str, event_date: str) -> list[dict[str, Any]] | None:
    del event_date
    if not GROQ_API_KEY:
        _log_provider_attempt(
            provider="groq",
            latency_ms=0,
            fact_count=0,
            fallback_reason="missing_api_key",
            input_token_count=_estimate_token_count(prompt),
            output_token_count=0,
        )
        return None

    body = {
        "model": GROQ_MODEL,
        "max_tokens": 2000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    return _call_provider(
        provider="groq",
        prompt=prompt,
        url=GROQ_URL,
        body=body,
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent": "rasputin-memory/0.9.1",
        },
        content_parser=_extract_openai_content,
        token_parser=_extract_openai_tokens,
    )


def _call_anthropic(prompt: str, event_date: str) -> list[dict[str, Any]] | None:
    del event_date
    if not ANTHROPIC_API_KEY:
        _log_provider_attempt(
            provider="anthropic",
            latency_ms=0,
            fact_count=0,
            fallback_reason="missing_api_key",
            input_token_count=_estimate_token_count(prompt),
            output_token_count=0,
        )
        return None

    body = {
        "model": FACT_EXTRACTION_MODEL,
        "max_tokens": 2000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    return _call_provider(
        provider="anthropic",
        prompt=prompt,
        url="https://api.anthropic.com/v1/messages",
        body=body,
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "User-Agent": "rasputin-memory/0.9.1",
        },
        content_parser=_extract_anthropic_content,
        token_parser=_extract_anthropic_tokens,
    )


def _extract_with_chain(prompt: str, event_date: str, starting_provider: str) -> list[dict[str, Any]]:
    callers: dict[str, Callable[[str, str], list[dict[str, Any]] | None]] = {
        "local_vllm": _call_local_vllm,
        "cerebras": _call_cerebras,
        "groq": _call_groq,
        "anthropic": _call_anthropic,
    }
    for provider in _provider_sequence(starting_provider):
        facts = callers[provider](prompt, event_date)
        if facts is not None:
            return facts
    raise RuntimeError("all extractors failed")


def _provider_sequence(starting_provider: str) -> list[str]:
    normalized = "anthropic" if starting_provider == "anthropic_only" else starting_provider
    if normalized not in _PROVIDER_ORDER:
        normalized = "local_vllm"
    start_index = _PROVIDER_ORDER.index(normalized)
    return list(_PROVIDER_ORDER[start_index:] + _PROVIDER_ORDER[:start_index])


def _call_provider(
    provider: str,
    prompt: str,
    url: str,
    body: dict[str, Any],
    timeout: int,
    headers: dict[str, str],
    content_parser: Callable[[dict[str, Any]], str],
    token_parser: Callable[[dict[str, Any], str, str], tuple[int, int]],
) -> list[dict[str, Any]] | None:
    backoffs = (1.0, 2.0)
    input_tokens = _estimate_token_count(prompt)

    for attempt_index, backoff_seconds in enumerate(backoffs, start=1):
        started_at = time.perf_counter()
        try:
            encoded = json.dumps(body).encode()
            req = request.Request(url, data=encoded, method="POST")
            for header_name, header_value in headers.items():
                req.add_header(header_name, header_value)

            with request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())

            content = _strip_thinking_tags(content_parser(data))
            input_tokens, output_tokens = token_parser(data, prompt, content)
            facts = _parse_extraction_response(content)
            _log_provider_attempt(
                provider=provider,
                latency_ms=round((time.perf_counter() - started_at) * 1000),
                fact_count=len(facts),
                fallback_reason=None,
                input_token_count=input_tokens,
                output_token_count=output_tokens,
            )
            return facts
        except Exception as exc:  # noqa: BLE001
            fallback_reason, is_retryable = _classify_provider_failure(exc, attempt_index)
            _log_provider_attempt(
                provider=provider,
                latency_ms=round((time.perf_counter() - started_at) * 1000),
                fact_count=0,
                fallback_reason=fallback_reason,
                input_token_count=input_tokens,
                output_token_count=0,
            )
            if is_retryable and attempt_index < len(backoffs):
                time.sleep(backoff_seconds)
                continue
            if is_retryable:
                time.sleep(backoff_seconds)
            return None

    return None


def _extract_openai_content(data: dict[str, Any]) -> str:
    return str(data["choices"][0]["message"]["content"]).strip()


def _extract_anthropic_content(data: dict[str, Any]) -> str:
    return str(data["content"][0]["text"]).strip()


def _extract_openai_tokens(data: dict[str, Any], prompt: str, content: str) -> tuple[int, int]:
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", _estimate_token_count(prompt))), int(
            usage.get("completion_tokens", _estimate_token_count(content))
        )
    return _estimate_token_count(prompt), _estimate_token_count(content)


def _extract_anthropic_tokens(data: dict[str, Any], prompt: str, content: str) -> tuple[int, int]:
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", _estimate_token_count(prompt))), int(
            usage.get("output_tokens", _estimate_token_count(content))
        )
    return _estimate_token_count(prompt), _estimate_token_count(content)


def _classify_provider_failure(exc: Exception, attempt_index: int) -> tuple[str, bool]:
    if isinstance(exc, error.HTTPError):
        status = exc.code
        return f"http_{status}", 500 <= status < 600

    if isinstance(exc, (TimeoutError, socket.timeout)):
        return f"timeout_{attempt_index}", True

    if isinstance(exc, error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return f"timeout_{attempt_index}", True
        return "connection_error", True

    if isinstance(exc, (ConnectionError, ConnectionRefusedError, OSError)):
        return "connection_error", True

    if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, ValidationError)):
        return "response_parse_error", False

    return exc.__class__.__name__.lower(), False


def _strip_thinking_tags(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _estimate_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(re.findall(r"\S+", stripped))


def _log_provider_attempt(
    provider: str,
    latency_ms: int,
    fact_count: int,
    fallback_reason: str | None,
    input_token_count: int,
    output_token_count: int,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider_used": provider,
        "latency_ms": latency_ms,
        "fact_count": fact_count,
        "session_id": _CURRENT_EXTRACTION_SESSION_ID,
        "fallback_reason": fallback_reason,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
    }
    try:
        log_path = Path(_EXTRACTION_PROVIDER_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write extraction provider log: %s", exc)


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
            except (ValidationError, TypeError) as exc:
                logger.warning("Pydantic validation error, falling back to raw parse: %s", exc)

            if "facts" in parsed and isinstance(parsed["facts"], list):
                return [fact for fact in parsed["facts"] if isinstance(fact, dict) and fact.get("what")]

    arr_start = raw.find("[")
    arr_end = raw.rfind("]") + 1
    if arr_start >= 0 and arr_end > arr_start:
        try:
            arr = json.loads(raw[arr_start:arr_end])
            if isinstance(arr, list):
                return [fact for fact in arr if isinstance(fact, dict) and fact.get("what")]
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
