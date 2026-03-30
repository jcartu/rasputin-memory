#!/usr/bin/env python3
"""
Second Brain Enrichment Pipeline v2 — Importance-Only Fast Pass

Batches 10 chunks per model call for importance scoring + 1-line summary.
Skips the heavy entity/topic/sentiment extraction that made v1 slow.

96K chunks × 10-per-call = 9,600 calls. At ~3s/call = ~8 hours total.
At 500 chunks/run × 4 runs/night = 2000/night → done in ~7 weeks.

Usage: python3 enrich_second_brain.py [--batch-size 500] [--chunks-per-call 10] [--dry-run]
"""

import argparse
import json
import time
import sys
import os
from datetime import datetime, timezone

import requests

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
PROXY_URL = "http://localhost:11438/v1/chat/completions"
MODEL = "qwen3.5-122b-a10b"
LOG_DIR = os.path.expanduser("~/.openclaw/workspace/memory/enrichment")

BATCH_PROMPT = """Score these text chunks on importance (1-10) and write a 1-sentence summary for each.

Importance guide:
8-10: Personal health decisions, business strategy, financial data, family/relationship info, legal/license matters, key contacts
5-7: Technical decisions, project milestones, useful research, relationship context, travel plans
1-4: Casual greetings, small talk, weather reports, generic links, "ok"/"thanks" messages, bot spam

{chunks_text}

Return ONLY a JSON array, one object per chunk:
[{{"id": 1, "importance": 7, "summary": "Discussion about Curacao license renewal timeline"}}, ...]

Rules:
- One object per chunk, in order
- importance must be integer 1-10
- summary must be 1 sentence, max 15 words
- Return ONLY the JSON array, no markdown fences"""


def qdrant_scroll(offset=None, limit=50):
    """Scroll through un-enriched vectors."""
    payload = {
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
        "filter": {"must_not": [{"key": "enriched", "match": {"value": True}}]},
    }
    if offset:
        payload["offset"] = offset

    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()["result"]
    return data.get("points", []), data.get("next_page_offset")


def qdrant_batch_update(updates):
    """Batch update multiple points' payloads."""
    for point_id, enrichment_data in updates:
        payload = {
            "points": [point_id],
            "payload": {
                "enriched": True,
                "importance_score": enrichment_data.get("importance", 0),
                "summary": enrichment_data.get("summary", ""),
                "enriched_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/payload", json=payload, timeout=10)
        r.raise_for_status()


def call_model_batch(chunks_with_text):
    """Send a batch of chunks to the model for importance scoring."""
    chunks_text = ""
    for i, (point_id, text) in enumerate(chunks_with_text, 1):
        # Truncate individual chunks to keep total prompt manageable
        truncated = text[:800] if len(text) > 800 else text
        truncated = truncated.replace("\n", " ").strip()
        chunks_text += f"Chunk {i}: {truncated}\n\n"

    prompt = BATCH_PROMPT.format(chunks_text=chunks_text)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.1,
    }

    r = requests.post(PROXY_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=90)
    r.raise_for_status()

    resp = r.json()
    msg = resp.get("choices", [{}])[0].get("message", {})
    content = msg.get("content")
    reasoning = msg.get("reasoning", "") or ""

    # First try direct JSON from content
    if content:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            results = json.loads(cleaned)
            if isinstance(results, list):
                return results
        except (json.JSONDecodeError, AttributeError):
            pass

    # For thinking models, extract decisions from reasoning output
    if reasoning:
        import re

        # Look for "Selected Importance: X" and "Selected Summary: ..." patterns
        imp_match = re.search(r"Selected\s+Importance\s*:\s*(\d+)", reasoning, re.IGNORECASE)
        sum_match = re.search(r'Selected\s+Summary\s*:\s*["\']([^"\']+)["\']', reasoning, re.IGNORECASE)

        if imp_match and sum_match:
            # Try to get the id from context (usually 1 for single chunk)
            return [{"id": 1, "importance": int(imp_match.group(1)), "summary": sum_match.group(1)}]

        # Fallback: look for JSON structure pattern with actual values
        json_match = re.search(r"JSON Structure:\s*(\[.*?\])", reasoning, re.DOTALL)
        if json_match:
            try:
                results = json.loads(json_match.group(1))
                if isinstance(results, list):
                    return results
            except json.JSONDecodeError:
                pass

    return None


def run_enrichment(batch_size=500, chunks_per_call=10, dry_run=False):
    """Main enrichment loop with batching."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    total_processed = 0
    total_enriched = 0
    total_failed = 0
    total_skipped = 0
    start_time = time.time()

    # Get remaining count
    try:
        count_r = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
            json={"filter": {"must_not": [{"key": "enriched", "match": {"value": True}}]}},
            timeout=10,
        )
        remaining = count_r.json()["result"]["count"]
    except:
        remaining = "unknown"

    print(f"[ENRICH v2] Starting. Batch: {batch_size}, Per-call: {chunks_per_call}. Remaining: {remaining}")

    # Collect all points for this batch
    all_points = []
    offset = None
    while len(all_points) < batch_size:
        fetch_size = min(500, batch_size - len(all_points))
        points, offset = qdrant_scroll(offset=offset, limit=fetch_size)
        if not points:
            break
        all_points.extend(points)
        if not offset:
            break

    if not all_points:
        print("[ENRICH v2] No un-enriched chunks found. All done!")
        return

    print(f"[ENRICH v2] Fetched {len(all_points)} chunks to process")

    # Process in batches of chunks_per_call
    for i in range(0, len(all_points), chunks_per_call):
        batch = all_points[i : i + chunks_per_call]
        chunks_with_text = []

        for point in batch:
            text = point.get("payload", {}).get("text", "")
            if not text or len(text.strip()) < 20:
                total_skipped += 1
                if not dry_run:
                    qdrant_batch_update([(point["id"], {"importance": 0, "summary": "too_short"})])
                continue
            chunks_with_text.append((point["id"], text))

        if not chunks_with_text:
            continue

        if dry_run:
            print(f"  [DRY] Batch {i // chunks_per_call + 1}: {len(chunks_with_text)} chunks")
            total_processed += len(chunks_with_text)
            continue

        try:
            results = call_model_batch(chunks_with_text)

            if results and len(results) >= len(chunks_with_text):
                updates = []
                for j, (point_id, _text) in enumerate(chunks_with_text):
                    if j < len(results):
                        r = results[j]
                        importance = r.get("importance", 0)
                        summary = r.get("summary", "")
                        updates.append((point_id, {"importance": importance, "summary": summary}))
                        total_enriched += 1

                qdrant_batch_update(updates)
                batch_num = i // chunks_per_call + 1
                avg_imp = sum(r.get("importance", 0) for r in results[: len(chunks_with_text)]) / len(chunks_with_text)
                print(f"  [OK] Batch {batch_num}: {len(chunks_with_text)} chunks, avg importance={avg_imp:.1f}")
            else:
                # Partial or failed parse — mark as enriched to avoid re-processing
                for point_id, _ in chunks_with_text:
                    qdrant_batch_update([(point_id, {"importance": 0, "summary": "parse_error"})])
                total_failed += len(chunks_with_text)
                print(
                    f"  [FAIL] Batch {i // chunks_per_call + 1}: Parse error ({len(results) if results else 0} results for {len(chunks_with_text)} chunks)"
                )

        except Exception as e:
            total_failed += len(chunks_with_text)
            print(f"  [ERROR] Batch {i // chunks_per_call + 1}: {str(e)[:100]}")

        total_processed += len(chunks_with_text)
        time.sleep(0.3)  # Brief pause between batches

    elapsed = time.time() - start_time
    rate = total_processed / elapsed if elapsed > 0 else 0

    summary = (
        f"[ENRICH v2] Done. "
        f"Processed: {total_processed}, Enriched: {total_enriched}, "
        f"Failed: {total_failed}, Skipped: {total_skipped}. "
        f"Time: {elapsed:.1f}s ({rate:.1f} chunks/s). "
        f"Remaining: ~{int(remaining) - total_processed if isinstance(remaining, int) else '?'}"
    )
    print(summary)

    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {summary}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich second brain (importance-only fast pass)")
    parser.add_argument("--batch-size", type=int, default=500, help="Total chunks per run")
    parser.add_argument("--chunks-per-call", type=int, default=10, help="Chunks per model call")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    run_enrichment(batch_size=args.batch_size, chunks_per_call=args.chunks_per_call, dry_run=args.dry_run)
