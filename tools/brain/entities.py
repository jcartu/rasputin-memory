from __future__ import annotations

import json
import os
import re
import time

from brain import _state

_known_entities_cache = None
_known_entities_cache_ts = 0.0
KNOWN_ENTITIES_CACHE_TTL_SECONDS = 5 * 60
KNOWN_PERSONS: set[str] = set()
KNOWN_ORGS: set[str] = set()
KNOWN_PROJECTS: set[str] = set()


def _load_known_entities() -> tuple[set[str], set[str], set[str]]:
    global _known_entities_cache
    global _known_entities_cache_ts
    global KNOWN_PERSONS
    global KNOWN_ORGS
    global KNOWN_PROJECTS

    now = time.time()
    if _known_entities_cache and (now - _known_entities_cache_ts) < KNOWN_ENTITIES_CACHE_TTL_SECONDS:
        return _known_entities_cache

    entities_path = _state.KNOWN_ENTITIES_PATH
    if not os.path.isabs(entities_path):
        entities_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", entities_path)
    try:
        with open(entities_path) as f:
            data = json.load(f)
        KNOWN_PERSONS = set(data.get("persons", []))
        KNOWN_ORGS = set(data.get("organizations", []))
        KNOWN_PROJECTS = set(data.get("projects", []))
        _known_entities_cache = (KNOWN_PERSONS, KNOWN_ORGS, KNOWN_PROJECTS)
        _known_entities_cache_ts = now
        return _known_entities_cache
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _state.logger.warning("Known entities config missing/invalid (%s), using empty sets", e)
        KNOWN_PERSONS = set()
        KNOWN_ORGS = set()
        KNOWN_PROJECTS = set()
        _known_entities_cache = (KNOWN_PERSONS, KNOWN_ORGS, KNOWN_PROJECTS)
        _known_entities_cache_ts = now
        return _known_entities_cache


def extract_entities_fast(text: str) -> list[tuple[str, str]]:
    extracted = []
    seen = set()

    known_persons, known_orgs, known_projects = _load_known_entities()
    text_lower = text.lower()

    for name in known_persons:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", text_lower) and name not in seen:
            seen.add(name)
            extracted.append((name, "Person"))

    for name in known_orgs:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", text_lower) and name not in seen:
            seen.add(name)
            extracted.append((name, "Organization"))

    for name in known_projects:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", text_lower) and name not in seen:
            seen.add(name)
            extracted.append((name, "Project"))

    for match in re.finditer(r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)+)\b", text):
        name = match.group(1)
        if name not in seen and len(name) > 4:
            seen.add(name)
            extracted.append((name, "Person"))

    _latin_stop = {"The", "This", "That", "What", "When", "Where", "Session", "Unknown", "None"}
    for match in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b", text):
        name = match.group(1)
        if name not in seen and len(name) > 4 and name.split()[0] not in _latin_stop:
            seen.add(name)
            extracted.append((name, "Person"))

    return extracted
