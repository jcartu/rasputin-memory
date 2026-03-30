#!/usr/bin/env python3

import json
import os
import sys

import requests

API_URL = os.environ.get("MEMORY_API_URL", "http://localhost:7777").rstrip("/")
DEFAULT_TIMEOUT = 30
TRIGGER_WORD_PATTERN = r"\b\w{3,}\b"
LEGACY_COMMIT_URL = "http://localhost:7777/commit"


def _request(method: str, path: str, *, params=None, payload=None, timeout: int = DEFAULT_TIMEOUT):
    url = f"{API_URL}{path}"
    if method == "GET":
        response = requests.get(url, params=params, timeout=timeout)
    else:
        response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def recall(query: str, limit: int = 10, expand: bool = True):
    return _request("GET", "/search", params={"q": query, "limit": limit, "expand": str(expand).lower()})


def commit(text: str, source: str = "conversation", importance: int = 60, metadata=None):
    payload = {"text": text, "source": source, "importance": importance, "metadata": metadata}
    return _request("POST", "/commit", payload=payload)


def deep(topic: str, limit: int = 20):
    return recall(f"{topic} details history decisions research email", limit=limit)


def whois(name: str, limit: int = 10):
    return recall(f"who is {name} profile relationships history", limit=limit)


def challenge(statement: str, limit: int = 8):
    return recall(f"{statement} contradictions risks alternatives", limit=limit)


def briefing(limit: int = 10):
    return recall("urgent important action required meeting appointment deadline payment", limit=limit)


def _print_search_results(result):
    for item in result.get("results", []):
        score = item.get("score", 0)
        text = item.get("text", "").replace("\n", " ")[:200]
        source = item.get("source", "")
        print(f"[{score:.3f}] ({source}) {text}")


def _usage():
    print("Usage: memory_engine.py <command> [args]")
    print("Commands: recall, commit, deep, whois, challenge, briefing")


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        _usage()
        return 1

    command = args[0]
    text = " ".join(args[1:]).strip()

    try:
        if command == "recall":
            if not text:
                _usage()
                return 1
            _print_search_results(recall(text))
            return 0

        if command == "commit":
            if not text:
                _usage()
                return 1
            print(json.dumps(commit(text), indent=2, ensure_ascii=False))
            return 0

        if command == "deep":
            if not text:
                _usage()
                return 1
            _print_search_results(deep(text))
            return 0

        if command == "whois":
            if not text:
                _usage()
                return 1
            _print_search_results(whois(text))
            return 0

        if command == "challenge":
            if not text:
                _usage()
                return 1
            _print_search_results(challenge(text))
            return 0

        if command == "briefing":
            _print_search_results(briefing())
            return 0

        _usage()
        return 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
