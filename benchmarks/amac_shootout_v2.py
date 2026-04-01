#!/usr/bin/env python3
"""A-MAC Shootout v2: Fix for Qwen3.5 thinking models.
Uses chat_template_kwargs={'enable_thinking': False} to suppress chain-of-thought.
Falls back to stripping <think>...</think> blocks and checking reasoning field."""

import requests
import json
import time
import re
import statistics
import math
from pathlib import Path

# Paths
RESULTS_PATH = Path("/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/amac_shootout_results.jsonl")
SUMMARY_PATH = Path("/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/amac_shootout_summary.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_3DIM = 'You are a memory quality evaluator. Score this memory on three criteria (0-10): Relevance (useful for future retrieval?), Novelty (genuinely new info?), Specificity (concrete retrievable details?). Return ONLY JSON: {"relevance":N,"novelty":N,"specificity":N,"composite":N} where composite = (R+N+S)/3. No explanation.'

SYSTEM_5DIM = 'You are a memory quality gate for an AI agent serving a CEO running an iGaming business ($2.5M/mo revenue), tracking geopolitics, managing AI infrastructure (dual RTX PRO 6000 + RTX 5090), and personal/family context. Score on 5 criteria (0-10): RELEVANCE (future retrieval value: 0=filler, 5=useful, 10=critical), NOVELTY (new info: 0=obvious, 10=first mention), SPECIFICITY (concrete details: 0=vague, 10=self-contained fact), ACTIONABILITY (future need: 0=expired, 10=critical deadline), TEMPORAL (longevity: 0=expired, 10=permanent). COMPOSITE = R*0.30 + N*0.20 + S*0.20 + A*0.15 + T*0.15. Return ONLY JSON: {"relevance":N,"novelty":N,"specificity":N,"actionability":N,"temporal":N,"composite":N}'

VARIANT_A = {
    "name": "35B Generic",
    "url": "http://localhost:11436/v1/chat/completions",
    "model": "qwen3.5:35b",
    "system": SYSTEM_3DIM,
    "max_tokens": 300,
}
VARIANT_B = {
    "name": "122B Generic",
    "url": "http://localhost:11435/v1/chat/completions",
    "model": "qwen3.5-122b-a10b",
    "system": SYSTEM_3DIM,
    "max_tokens": 300,
}
VARIANT_C = {
    "name": "122B Domain-Specific",
    "url": "http://localhost:11435/v1/chat/completions",
    "model": "qwen3.5-122b-a10b",
    "system": SYSTEM_5DIM,
    "max_tokens": 300,
}


def load_existing_samples():
    """Load point IDs and text from existing results JSONL."""
    samples = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            samples.append(
                {
                    "id": r["id"],
                    "text": r.get("text_preview", ""),
                }
            )
    print(f"Loaded {len(samples)} samples from existing results")
    return samples


def get_full_texts_from_qdrant(ids):
    """Fetch full text from Qdrant for the given point IDs."""
    print(f"Fetching full texts for {len(ids)} points from Qdrant...")
    id_to_text = {}
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        resp = requests.post(
            "http://localhost:6333/collections/memories_v2/points",
            json={"ids": batch, "with_payload": True, "with_vector": False},
            timeout=30,
        )
        resp.raise_for_status()
        for point in resp.json()["result"]:
            pid = point["id"]
            payload = point.get("payload", {})
            text = None
            for field in ["text", "content", "memory", "body", "value"]:
                if field in payload and payload[field]:
                    text = str(payload[field])
                    break
            if text is None:
                parts = [str(v) for v in payload.values() if isinstance(v, str) and v]
                text = " | ".join(parts) if parts else str(payload)
            id_to_text[pid] = text
    print(f"Fetched {len(id_to_text)} full texts")
    return id_to_text


def parse_json_response(text):
    """Extract JSON from model response, handling <think>...</think> blocks."""
    if not text:
        return None

    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try direct parse of cleaned text
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Regex: find JSON with "relevance" key in cleaned text
    m = re.search(r'\{[^{}]*"relevance"[^{}]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Fallback: try full original text
    m = re.search(r'\{[^{}]*"relevance"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Last resort: extract key:value pairs
    result = {}
    for key in ["relevance", "novelty", "specificity", "actionability", "temporal", "composite"]:
        match = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', cleaned or text)
        if match:
            result[key] = float(match.group(1))
    return result if result else None


def call_model(variant, memory_text):
    user_msg = f"Memory text: {memory_text[:1000]} /no_think"
    payload = {
        "model": variant["model"],
        "messages": [{"role": "system", "content": variant["system"]}, {"role": "user", "content": user_msg}],
        "temperature": 0.1,
        "max_tokens": variant["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.time()
    try:
        resp = requests.post(variant["url"], json=payload, timeout=180)
        resp.raise_for_status()
        elapsed_ms = (time.time() - start) * 1000
        msg = resp.json()["choices"][0]["message"]

        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""

        # Try content first (strip <think> blocks), then reasoning field
        parsed = parse_json_response(content)
        if parsed is None and reasoning:
            parsed = parse_json_response(reasoning)

        raw_text = (content or reasoning)[:300]
        return parsed, elapsed_ms, raw_text
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        print(f"  ERROR calling {variant['name']}: {e}")
        return None, elapsed_ms, str(e)


def get_samples_from_qdrant():
    """Fetch 200 memory samples directly from Qdrant."""
    print("Fetching 200 memories from Qdrant...")
    resp = requests.post(
        "http://localhost:6333/collections/memories_v2/points/scroll",
        json={"limit": 200, "with_payload": True, "with_vector": False},
        timeout=30,
    )
    resp.raise_for_status()
    points = resp.json()["result"]["points"]
    print(f"Got {len(points)} points")
    samples = []
    for point in points:
        pid = point.get("id")
        payload = point.get("payload", {})
        text = None
        for field in ["text", "content", "memory", "body", "value"]:
            if field in payload and payload[field]:
                text = str(payload[field])
                break
        if text is None:
            parts = [str(v) for v in payload.values() if isinstance(v, str) and v]
            text = " | ".join(parts) if parts else str(payload)
        samples.append({"id": pid, "text": text})
    return samples


def main():
    # Always fetch fresh from Qdrant
    samples = get_samples_from_qdrant()

    results = []
    stats = {
        "a": {"composites": [], "latencies": [], "failures": 0},
        "b": {"composites": [], "latencies": [], "failures": 0},
        "c": {"composites": [], "latencies": [], "failures": 0},
    }

    print(f"\nProcessing {len(samples)} memories (v2 — thinking disabled)...")

    with open(RESULTS_PATH, "w") as f_out:
        for i, sample in enumerate(samples):
            pid = sample["id"]
            text = sample["text"]

            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(samples)}] id={pid} text_len={len(text)}")

            record = {"id": pid, "text_preview": text[:120], "variants": {}}

            for key, variant in [("a", VARIANT_A), ("b", VARIANT_B), ("c", VARIANT_C)]:
                parsed, latency_ms, raw = call_model(variant, text)
                composite = None
                if parsed:
                    composite = parsed.get("composite")
                    if composite is not None:
                        composite = float(composite)

                stats[key]["latencies"].append(latency_ms)
                if composite is not None:
                    stats[key]["composites"].append(composite)
                else:
                    stats[key]["failures"] += 1

                record["variants"][key] = {
                    "name": variant["name"],
                    "scores": parsed,
                    "composite": composite,
                    "latency_ms": round(latency_ms, 1),
                    "parse_ok": parsed is not None,
                }

                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"  {variant['name']}: composite={composite} latency={latency_ms:.0f}ms parse_ok={parsed is not None}"
                    )

            results.append(record)
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

    print("\nComputing summary...")

    def variant_summary(key):
        c = stats[key]["composites"]
        lats = stats[key]["latencies"]
        total = len(c) + stats[key]["failures"]
        if not c:
            return {
                "mean_composite": None,
                "median": None,
                "stdev": None,
                "reject_rate_at_4": None,
                "mean_latency_ms": round(statistics.mean(lats), 1) if lats else None,
                "parse_failures": stats[key]["failures"],
                "parse_success_rate": 0.0,
            }
        return {
            "mean_composite": round(statistics.mean(c), 3),
            "median": round(statistics.median(c), 3),
            "stdev": round(statistics.stdev(c) if len(c) > 1 else 0, 3),
            "reject_rate_at_4": round(sum(1 for x in c if x < 4) / len(c), 3),
            "mean_latency_ms": round(statistics.mean(lats), 1),
            "parse_failures": stats[key]["failures"],
            "parse_success_rate": round(len(c) / total, 3) if total else 0,
        }

    def correlation(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) < 2:
            return None
        xs2 = [p[0] for p in pairs]
        ys2 = [p[1] for p in pairs]
        mx, my = statistics.mean(xs2), statistics.mean(ys2)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den = math.sqrt(sum((x - mx) ** 2 for x in xs2) * sum((y - my) ** 2 for y in ys2))
        return round(num / den, 4) if den else None

    a_scores = [r["variants"]["a"]["composite"] for r in results]
    b_scores = [r["variants"]["b"]["composite"] for r in results]
    c_scores = [r["variants"]["c"]["composite"] for r in results]

    # Top disagreements (largest A vs C delta)
    disagreements = []
    for r in results:
        a = r["variants"]["a"]["composite"]
        c_val = r["variants"]["c"]["composite"]
        b = r["variants"]["b"]["composite"]
        if a is not None and c_val is not None:
            delta = abs(a - c_val)
            disagreements.append((delta, r["text_preview"], a, b, c_val))
    disagreements.sort(reverse=True)
    top_disagreements = [
        {"memory_preview": d[1], "a_score": d[2], "b_score": d[3], "c_score": d[4]} for d in disagreements[:5]
    ]

    summary = {
        "total_memories": len(samples),
        "variant_a_35b": variant_summary("a"),
        "variant_b_122b": variant_summary("b"),
        "variant_c_122b_domain": variant_summary("c"),
        "agreement": {
            "a_vs_b_correlation": correlation(a_scores, b_scores),
            "b_vs_c_correlation": correlation(b_scores, c_scores),
            "a_vs_c_correlation": correlation(a_scores, c_scores),
        },
        "interesting_disagreements": top_disagreements,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Done!")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    main()
