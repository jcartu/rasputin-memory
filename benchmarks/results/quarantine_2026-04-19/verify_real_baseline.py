#!/usr/bin/env python3
"""Check if 59c0a369...-locomo-production.json is the real baseline matching baseline.log."""
from __future__ import annotations
import json
from collections import defaultdict

REAL = "benchmarks/results/59c0a369b4296182cf69f11d74beda56e00e14eb-locomo-production.json"
CKPT = "benchmarks/results/locomo-leaderboard-checkpoint.json"

def score(path):
    print(f"\n=== {path} ===")
    with open(path) as f:
        d = json.load(f)
    print(f"top-level keys: {list(d.keys())[:10]}")
    rows = d.get("results") or d.get("rows") or d.get("data") or []
    if not rows and isinstance(d, list):
        rows = d
    # If structured differently (e.g. nested per-conv), flatten
    if rows and isinstance(rows, dict):
        flat = []
        for k, v in rows.items():
            if isinstance(v, list):
                flat.extend(v)
        rows = flat
    print(f"rows: {len(rows)}")
    if rows:
        print(f"sample row keys: {list(rows[0].keys())[:15]}")

    by_conv_nonadv = defaultdict(lambda: [0, 0])
    overall_nonadv = [0, 0]
    overall_all = [0, 0]
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cat = r.get("cat_name") or r.get("category") or "?"
        conv = r.get("conv_id") or r.get("conversation_id") or "?"
        correct = 1 if r.get("correct") else 0
        by_cat[cat][0] += correct; by_cat[cat][1] += 1
        overall_all[0] += correct; overall_all[1] += 1
        if "adv" not in str(cat).lower():
            by_conv_nonadv[conv][0] += correct
            by_conv_nonadv[conv][1] += 1
            overall_nonadv[0] += correct
            overall_nonadv[1] += 1
    if overall_all[1]:
        print(f"overall all: {overall_all[0]}/{overall_all[1]} = {100*overall_all[0]/overall_all[1]:.2f}%")
    if overall_nonadv[1]:
        print(f"overall non-adv: {overall_nonadv[0]}/{overall_nonadv[1]} = {100*overall_nonadv[0]/overall_nonadv[1]:.2f}%")
    print("by cat:")
    for cat, (c, t) in sorted(by_cat.items()):
        print(f"  {cat:<20} {c}/{t} = {100*c/t:.2f}%" if t else f"  {cat}: empty")
    print("by conv (non-adv):")
    for conv, (c, t) in sorted(by_conv_nonadv.items()):
        print(f"  {conv:<15} {c}/{t} = {100*c/t:.2f}%")

score(REAL)
score(CKPT)
