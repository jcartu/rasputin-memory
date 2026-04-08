#!/usr/bin/env python3
"""Compare LoCoMo-Plus baseline vs constraint-enabled benchmark runs."""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

BASELINE_CP = RESULTS_DIR / "locomo-plus-checkpoint.json"
CONSTRAINTS_CP = RESULTS_DIR / "locomo-plus-constraints-checkpoint.json"
OUTPUT = RESULTS_DIR / "locomo-plus-comparison.md"

CATEGORIES = ["single-hop", "multi-hop", "temporal", "common-sense", "adversarial", "Cognitive"]


def load_results(path):
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("results", {})


def category_stats(results):
    by_cat = defaultdict(list)
    for info in results.values():
        by_cat[info.get("category", "?")].append(info.get("judge_score", 0))
    return by_cat


def main():
    baseline = load_results(BASELINE_CP)
    constraints = load_results(CONSTRAINTS_CP)

    if not baseline:
        print(f"No baseline results at {BASELINE_CP}")
        sys.exit(1)
    if not constraints:
        print(f"No constraint results at {CONSTRAINTS_CP}")
        sys.exit(1)

    b_cats = category_stats(baseline)
    c_cats = category_stats(constraints)

    b_all = [s for scores in b_cats.values() for s in scores]
    c_all = [s for scores in c_cats.values() for s in scores]
    b_overall = sum(b_all) / len(b_all) * 100
    c_overall = sum(c_all) / len(c_all) * 100

    factual = ["single-hop", "multi-hop", "temporal", "common-sense"]
    b_factual = [s for cat in factual for s in b_cats.get(cat, [])]
    c_factual = [s for cat in factual for s in c_cats.get(cat, [])]
    b_factual_avg = sum(b_factual) / len(b_factual) * 100 if b_factual else 0
    c_factual_avg = sum(c_factual) / len(c_factual) * 100 if c_factual else 0

    b_cog = b_cats.get("Cognitive", [])
    c_cog = c_cats.get("Cognitive", [])
    b_cog_avg = sum(b_cog) / len(b_cog) * 100 if b_cog else 0
    c_cog_avg = sum(c_cog) / len(c_cog) * 100 if c_cog else 0

    lines = [
        "# LoCoMo-Plus: Baseline vs Constraint-Enabled",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Baseline samples:** {len(baseline)}",
        f"**Constraint samples:** {len(constraints)}",
        "",
        "## Summary",
        "",
        "| Metric | Baseline | Constraints | Delta |",
        "|--------|----------|-------------|-------|",
        f"| **Overall** | {b_overall:.1f}% | {c_overall:.1f}% | {c_overall - b_overall:+.1f}% |",
        f"| **Factual (Level-1)** | {b_factual_avg:.1f}% | {c_factual_avg:.1f}% | {c_factual_avg - b_factual_avg:+.1f}% |",
        f"| **Cognitive (Level-2)** | {b_cog_avg:.1f}% | {c_cog_avg:.1f}% | {c_cog_avg - b_cog_avg:+.1f}% |",
        "",
        "## Per-Category Breakdown",
        "",
        "| Category | N | Baseline | Constraints | Delta |",
        "|----------|---|----------|-------------|-------|",
    ]

    for cat in CATEGORIES:
        b_scores = b_cats.get(cat, [])
        c_scores = c_cats.get(cat, [])
        n = max(len(b_scores), len(c_scores))
        b_avg = sum(b_scores) / len(b_scores) * 100 if b_scores else 0
        c_avg = sum(c_scores) / len(c_scores) * 100 if c_scores else 0
        delta = c_avg - b_avg
        lines.append(f"| {cat} | {n} | {b_avg:.1f}% | {c_avg:.1f}% | {delta:+.1f}% |")

    lines.extend(
        [
            "",
            "## Regression Check",
            "",
        ]
    )
    regression = False
    for cat in factual:
        b_scores = b_cats.get(cat, [])
        c_scores = c_cats.get(cat, [])
        if not b_scores or not c_scores:
            continue
        b_avg = sum(b_scores) / len(b_scores) * 100
        c_avg = sum(c_scores) / len(c_scores) * 100
        if c_avg < b_avg - 2.0:
            lines.append(
                f"⚠️ **{cat}**: regressed {c_avg - b_avg:.1f}% (baseline {b_avg:.1f}% → constraints {c_avg:.1f}%)"
            )
            regression = True

    if not regression:
        lines.append("✅ No factual category regressed by more than 2%.")

    report = "\n".join(lines) + "\n"
    with open(OUTPUT, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
