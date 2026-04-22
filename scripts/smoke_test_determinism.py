#!/usr/bin/env python3
"""Determinism smoke test for local vLLM extraction.

Required by user authorization to kill 122B: "32B produces deterministic
extractions in smoke test" (post-exit-gate-redeploy-plan.md Step 1).

Runs N identical extraction calls against LOCAL_VLLM_URL with fixed seed,
asserts all outputs are byte-identical.

Exit 0 on pass, 1 on fail.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from urllib.error import URLError

ENDPOINT = os.environ.get("LOCAL_VLLM_URL", "http://localhost:11437/v1/chat/completions")
MODEL = os.environ.get("LOCAL_VLLM_MODEL", "qwen3-32b-awq")
SEED = int(os.environ.get("LOCAL_VLLM_SEED", "42"))
N_CALLS = int(os.environ.get("SMOKE_N_CALLS", "10"))

FIXED_PROMPT = (
    "Extract 3-5 discrete facts from this conversation turn. Event date: 2026-04-21.\n\n"
    "Alice: I joined Google as a staff engineer in March 2020.\n"
    "Bob: Congrats! Where's the office?\n"
    "Alice: Mountain View. Lunch is free.\n\n"
    'Return JSON: {"facts": [{"fact": str, "entities": [str], "confidence": float}]}'
)


def call_once() -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": FIXED_PROMPT}],
        "temperature": 0.0,
        "seed": SEED,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    return content


def main() -> int:
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model: {MODEL}")
    print(f"Seed: {SEED}")
    print(f"Calls: {N_CALLS}")
    print("-" * 60)
    outputs: list[str] = []
    hashes: list[str] = []
    for i in range(N_CALLS):
        t0 = time.time()
        try:
            out = call_once()
        except (URLError, TimeoutError, OSError) as e:
            print(f"Call {i + 1}/{N_CALLS}: FAILED ({e})")
            return 1
        outputs.append(out)
        h = hashlib.sha256(out.encode("utf-8")).hexdigest()[:16]
        hashes.append(h)
        dt = time.time() - t0
        print(f"Call {i + 1}/{N_CALLS}: {dt:5.2f}s sha256={h} len={len(out)}")
    print("-" * 60)
    unique = set(hashes)
    if len(unique) == 1:
        print(f"PASS: all {N_CALLS} outputs identical (sha256={hashes[0]})")
        print(f"Sample output (first 300 chars):\n{outputs[0][:300]}")
        return 0
    print(f"FAIL: {len(unique)} distinct outputs across {N_CALLS} calls")
    for i, h in enumerate(hashes):
        print(f"  call {i + 1}: {h}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
