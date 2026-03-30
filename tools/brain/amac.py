from __future__ import annotations

import datetime as _dt_mod
import json
import os
import re
from typing import Any, Optional

from brain import _state


def _inc_metric(key: str, amount: float = 1) -> None:
    with _state._amac_metrics_lock:
        _state._amac_metrics[key] += amount


AMAC_PROMPT_TEMPLATE = """You are a memory quality filter for an AI agent. Score the following memory on 3 dimensions.
Return ONLY three integers separated by commas. No text, no explanation, no reasoning. Just three numbers like: 7,4,8

Relevance 0-10: Is this about AI infrastructure, business operations, technology, the user's domain of interest? (0=totally unrelated, 10=highly relevant)
Novelty 0-10: Does this add genuinely NEW, specific information? (0=generic platitude, 10=unique concrete fact)
Specificity 0-10: Is this a concrete verifiable fact with numbers/names/dates? (0=vague filler, 10=specific actionable data)

Examples:
"Things are going well." -> 0,1,0
"BTC went up today." -> 4,2,2
"BrandA DACH revenue hit €580K in Feb 2026, up 23% MoM." -> 10,9,10
"The user's family member had a medical procedure at a major hospital." -> 10,9,10

Memory: "{text}"

Output format: SCORES: R,N,S (three integers separated by commas, nothing else)
"""


def amac_score(text: str, retry: int = 2) -> Optional[tuple[float, float, float, float]]:
    prompt = AMAC_PROMPT_TEMPLATE.format(text=text[:800])

    for attempt in range(retry + 1):
        try:
            response = _state.requests.post(
                os.environ.get("AMAC_LLM_URL", _state.AMAC_LLM_URL),
                json={
                    "model": _state.AMAC_OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.05,
                    "max_tokens": 500,
                },
                timeout=_state.AMAC_TIMEOUT,
            )
            response.raise_for_status()
            raw = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            sentinel_match = re.search(r"SCORES:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
            if sentinel_match:
                sentinel_text = sentinel_match.group(1)
                sentinel_scores = re.findall(r"(?<!\d)(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})(?!\d)", sentinel_text)
                for score_triplet in sentinel_scores:
                    if all(0 <= int(value) <= 10 for value in score_triplet):
                        relevance = float(score_triplet[0])
                        novelty = float(score_triplet[1])
                        specificity = float(score_triplet[2])
                        composite = round((relevance + novelty + specificity) / 3, 2)
                        _state.logger.info(
                            "A-MAC parsed sentinel scores r=%s n=%s s=%s",
                            relevance,
                            novelty,
                            specificity,
                        )
                        return relevance, novelty, specificity, composite

            lines = raw.split("\n")
            all_triplets = []

            for line in lines:
                stripped = line.strip()
                scores = re.findall(r"(?<!\d)(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})(?!\d)", stripped)
                for score_triplet in scores:
                    if all(0 <= int(value) <= 10 for value in score_triplet):
                        all_triplets.append(score_triplet)

            if not all_triplets:
                _state.logger.warning(
                    "A-MAC could not parse score triplets from response fragment: %s", repr(raw[:200])
                )
                return None

            relevance = float(all_triplets[-1][0])
            novelty = float(all_triplets[-1][1])
            specificity = float(all_triplets[-1][2])
            composite = round((relevance + novelty + specificity) / 3, 2)
            _state.logger.info(
                "A-MAC parsed scores from triplet #%s: r=%s n=%s s=%s",
                len(all_triplets),
                relevance,
                novelty,
                specificity,
            )
            return relevance, novelty, specificity, composite

        except _state.requests.exceptions.Timeout:
            _state.logger.warning("A-MAC timeout; fail-open accepting commit")
            return None
        except Exception as error:
            if attempt < retry:
                _state.logger.warning("A-MAC scoring error on attempt %s, retrying: %s", attempt + 1, error)
                continue
            _state.logger.error("A-MAC scoring error; fail-open: %s", error)
            return None

    return None


def amac_gate(text: str, source: str = "unknown", force: bool = False) -> tuple[bool, str, dict[str, Any]]:
    if force:
        _inc_metric("bypassed")
        return True, "bypassed", {}

    if re.search(
        r"PIPELINE_TEST_|HEALTH_CHECK_|SYSTEM_DIAGNOSTIC_|memory health check|self-diagnostic",
        text,
        re.IGNORECASE,
    ):
        return True, "diagnostic_skip", {}

    scores = amac_score(text)

    if scores is None:
        _inc_metric("accepted")
        _inc_metric("timeout_accepts")
        return True, "timeout_failopen", {}

    relevance, novelty, specificity, composite = scores
    score_dict = {
        "relevance": relevance,
        "novelty": novelty,
        "specificity": specificity,
        "composite": composite,
    }

    _inc_metric("score_sum", composite)
    _inc_metric("score_count")

    if composite >= _state.AMAC_THRESHOLD:
        _inc_metric("accepted")
        return True, "accepted", score_dict

    _inc_metric("rejected")
    try:
        entry = {
            "ts": _dt_mod.datetime.now().isoformat(),
            "source": source,
            "scores": score_dict,
            "text": text[:200],
        }
        with open(_state.AMAC_REJECT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as log_error:
        _state.logger.error("A-MAC failed to write reject log: %s", log_error)
    return False, "rejected", score_dict
