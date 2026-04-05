#!/usr/bin/env python3
"""Failure taxonomy analysis for RASPUTIN benchmark results.

Takes a benchmark result JSON (with chunks stored) and produces:
- Retrieval metrics: Gold-in-any-chunk, Gold-in-top-5, Gold-in-top-10, MRR
- Failure taxonomy: retrieval_miss, retrieval_buried, generation_failure
- Per-category breakdown

Usage:
    python3 benchmarks/analyze_failures.py benchmarks/results/<hash>-locomo-production.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def gold_in_chunks(gold: str, chunks: list[str], threshold: float = 0.5) -> int | None:
    gold_tokens = set(TOKEN_RE.findall(gold.lower()))
    if not gold_tokens or len(gold_tokens) < 2:
        gold_lower = gold.lower().strip()
        for i, chunk in enumerate(chunks):
            if gold_lower in chunk.lower():
                return i
        return None

    for i, chunk in enumerate(chunks):
        chunk_tokens = set(TOKEN_RE.findall(chunk.lower()))
        overlap = gold_tokens & chunk_tokens
        if len(overlap) >= max(1, len(gold_tokens) * threshold):
            return i
    return None


def analyze(results: list[dict]) -> dict:
    cat_metrics: dict[str, dict] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "gold_in_any": 0,
            "gold_in_top5": 0,
            "gold_in_top10": 0,
            "retrieval_miss": 0,
            "retrieval_buried": 0,
            "generation_failure": 0,
            "mrr_sum": 0.0,
            "f1_sum": 0.0,
        }
    )

    for r in results:
        cat = r.get("cat_name") or CATEGORY_NAMES.get(r.get("category", 0), "unknown")
        m = cat_metrics[cat]
        m["total"] += 1
        if r.get("correct"):
            m["correct"] += 1

        m["f1_sum"] += r.get("f1", 0)

        chunks = r.get("chunks", [])
        gold = str(r.get("gold", ""))
        if not chunks:
            if not r.get("correct"):
                m["retrieval_miss"] += 1
            continue

        rank = gold_in_chunks(gold, chunks)

        if rank is not None:
            m["gold_in_any"] += 1
            m["mrr_sum"] += 1.0 / (rank + 1)
            if rank < 5:
                m["gold_in_top5"] += 1
            if rank < 10:
                m["gold_in_top10"] += 1

            if not r.get("correct"):
                if rank < 10:
                    m["generation_failure"] += 1
                else:
                    m["retrieval_buried"] += 1
        else:
            if not r.get("correct"):
                m["retrieval_miss"] += 1

    return dict(cat_metrics)


def print_report(cat_metrics: dict, output_path: str | None = None):
    lines = []

    def p(s=""):
        lines.append(s)

    p("# Failure Taxonomy Report")
    p()

    all_total = sum(m["total"] for m in cat_metrics.values())
    all_correct = sum(m["correct"] for m in cat_metrics.values())
    all_gold_any = sum(m["gold_in_any"] for m in cat_metrics.values())
    all_gold_5 = sum(m["gold_in_top5"] for m in cat_metrics.values())
    all_gold_10 = sum(m["gold_in_top10"] for m in cat_metrics.values())

    p(f"**Total:** {all_total} questions, {all_correct} correct ({all_correct / max(all_total, 1) * 100:.1f}%)")
    p(f"**Gold-in-ANY-chunk:** {all_gold_any}/{all_total} ({all_gold_any / max(all_total, 1) * 100:.1f}%)")
    p(f"**Gold-in-Top-5:** {all_gold_5}/{all_total} ({all_gold_5 / max(all_total, 1) * 100:.1f}%)")
    p(f"**Gold-in-Top-10:** {all_gold_10}/{all_total} ({all_gold_10 / max(all_total, 1) * 100:.1f}%)")
    p()

    p("## Retrieval Metrics by Category")
    p()
    p("| Category | Total | Accuracy | Gold-ANY | Gold-Top5 | Gold-Top10 | MRR |")
    p("|----------|-------|----------|----------|-----------|------------|-----|")
    for cat in sorted(cat_metrics.keys()):
        m = cat_metrics[cat]
        t = m["total"]
        p(
            f"| {cat} | {t} | {m['correct'] / max(t, 1) * 100:.1f}% | "
            f"{m['gold_in_any'] / max(t, 1) * 100:.0f}% | "
            f"{m['gold_in_top5'] / max(t, 1) * 100:.0f}% | "
            f"{m['gold_in_top10'] / max(t, 1) * 100:.0f}% | "
            f"{m['mrr_sum'] / max(t, 1):.3f} |"
        )

    p()
    p("## Failure Taxonomy (incorrect answers only)")
    p()
    p("| Category | Wrong | Retrieval Miss | Retrieval Buried | Generation Fail |")
    p("|----------|-------|----------------|------------------|-----------------|")
    for cat in sorted(cat_metrics.keys()):
        m = cat_metrics[cat]
        wrong = m["total"] - m["correct"]
        if wrong == 0:
            continue
        p(
            f"| {cat} | {wrong} | "
            f"{m['retrieval_miss']} ({m['retrieval_miss'] * 100 // max(wrong, 1)}%) | "
            f"{m['retrieval_buried']} ({m['retrieval_buried'] * 100 // max(wrong, 1)}%) | "
            f"{m['generation_failure']} ({m['generation_failure'] * 100 // max(wrong, 1)}%) |"
        )

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).write_text(report)
        print(f"\nSaved to {output_path}")

    return report


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result-json> [--output <path>]")
        sys.exit(1)

    result_path = sys.argv[1]
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    with open(result_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in file.")
        sys.exit(1)

    has_chunks = any(r.get("chunks") for r in results)
    if not has_chunks:
        print("WARNING: No chunks stored in results. Retrieval metrics will be inaccurate.")

    cat_metrics = analyze(results)
    print_report(cat_metrics, output_path)

    metrics_path = Path(result_path).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(cat_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
