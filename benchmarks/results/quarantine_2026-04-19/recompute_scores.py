#!/usr/bin/env python3
"""Recompute per-conv and overall non-adv score from v1.json and checkpoint.json."""
from __future__ import annotations
import json
from collections import defaultdict

for path in [
    "benchmarks/results/locomo-leaderboard-checkpoint.json",
    "benchmarks/results/locomo-leaderboard-v1.json",
]:
    print(f"\n=== {path} ===")
    with open(path) as f:
        d = json.load(f)
    rows = d.get("results", [])
    print(f"total rows: {len(rows)}")

    # category breakdown
    by_cat = defaultdict(lambda: [0, 0])  # [correct, total]
    by_conv = defaultdict(lambda: [0, 0])
    by_conv_nonadv = defaultdict(lambda: [0, 0])
    overall_nonadv = [0, 0]
    overall_all = [0, 0]

    for r in rows:
        cat = r.get("cat_name", r.get("category", "?"))
        conv = r.get("conv_id", "?")
        correct = 1 if r.get("correct") else 0
        by_cat[cat][0] += correct
        by_cat[cat][1] += 1
        by_conv[conv][0] += correct
        by_conv[conv][1] += 1
        overall_all[0] += correct
        overall_all[1] += 1
        if cat != "adversarial" and "adv" not in str(cat).lower():
            by_conv_nonadv[conv][0] += correct
            by_conv_nonadv[conv][1] += 1
            overall_nonadv[0] += correct
            overall_nonadv[1] += 1

    print(f"\nOverall all: {overall_all[0]}/{overall_all[1]} = {100*overall_all[0]/overall_all[1]:.2f}%")
    print(f"Overall non-adv: {overall_nonadv[0]}/{overall_nonadv[1]} = {100*overall_nonadv[0]/overall_nonadv[1]:.2f}%")
    print("\nBy category:")
    for cat, (c, t) in sorted(by_cat.items()):
        print(f"  {cat:<25} {c}/{t} = {100*c/t:.2f}%")
    print("\nBy conv (non-adv):")
    for conv, (c, t) in sorted(by_conv_nonadv.items()):
        print(f"  {conv:<15} {c}/{t} = {100*c/t:.2f}%")
