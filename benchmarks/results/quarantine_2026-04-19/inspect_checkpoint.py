"""Inspect the Phase A checkpoint structure to determine whether per-candidate
data (needed for the proof_norm diagnostic) is persisted.

Per RASPUTIN_NEXT_ACTION.md Step 1:
  Path A -> candidates persisted (fields like candidates/top_60/retrieved/sources)
  Path B -> no candidates persisted, need a targeted re-bench with SCORE_BREAKDOWN=1
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def describe_value(v, depth=0, max_depth=3):
    pad = "  " * depth
    if depth > max_depth:
        return f"{pad}<...>"
    if isinstance(v, dict):
        lines = [f"{pad}dict (keys={sorted(v.keys())})"]
        for k in sorted(v.keys())[:8]:
            lines.append(f"{pad}  .{k}: {type(v[k]).__name__}")
        return "\n".join(lines)
    if isinstance(v, list):
        if not v:
            return f"{pad}[] (empty)"
        first = v[0]
        if isinstance(first, dict):
            return f"{pad}list[dict] len={len(v)} first.keys={sorted(first.keys())[:20]}"
        return f"{pad}list[{type(first).__name__}] len={len(v)} first={first!r:.100}"
    return f"{pad}{type(v).__name__}={v!r:.120}"


def main(path: str) -> None:
    print(f"Loading {path} ...")
    with open(path) as f:
        ckpt = json.load(f)
    print(f"Top-level type: {type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {sorted(ckpt.keys())}")
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            if isinstance(v, list):
                print(f"  .{k}: list len={len(v)}")
            elif isinstance(v, dict):
                print(f"  .{k}: dict keys={sorted(v.keys())[:15]}")
            else:
                print(f"  .{k}: {type(v).__name__} = {v!r:.120}")

    results = None
    if isinstance(ckpt, dict):
        for key in ("results", "entries", "records", "questions"):
            if key in ckpt and isinstance(ckpt[key], list):
                results = ckpt[key]
                print(f"\nFound results under .{key} (len={len(results)})")
                break
    elif isinstance(ckpt, list):
        results = ckpt
        print(f"\nTop-level is a list, treating as results (len={len(results)})")

    if not results:
        print("No results list located. Inspecting raw top-level instead.")
        print(describe_value(ckpt, max_depth=2))
        return

    print(f"\nTotal result entries: {len(results)}")
    if not results:
        return

    # Survey keys across a sample
    key_counter: Counter[str] = Counter()
    list_dict_keys: dict[str, Counter[str]] = {}
    sample_n = min(200, len(results))
    for r in results[:sample_n]:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            key_counter[k] += 1
            if isinstance(v, list) and v and isinstance(v[0], dict):
                c = list_dict_keys.setdefault(k, Counter())
                for kk in v[0].keys():
                    c[kk] += 1

    print(f"\nKeys present in first {sample_n} entries (counts):")
    for k, cnt in key_counter.most_common():
        marker = " <-- candidate-list-looking" if k in list_dict_keys else ""
        print(f"  {k}: {cnt}{marker}")

    if list_dict_keys:
        print("\nFields that are list[dict] (possible candidate lists):")
        for field, sub in list_dict_keys.items():
            print(f"  .{field}: inner keys = {[k for k, _ in sub.most_common()]}")
    else:
        print("\nNo list[dict] fields found at top level of a result.")

    # Dump the first entry in detail
    r0 = results[0]
    print("\n=== First entry structure (truncated) ===")
    if isinstance(r0, dict):
        for k in sorted(r0.keys()):
            v = r0[k]
            if isinstance(v, list):
                if v and isinstance(v[0], dict):
                    print(f"  .{k}: list[dict] len={len(v)}")
                    print(f"      first item keys: {sorted(v[0].keys())[:30]}")
                    print(f"      first item sample: {json.dumps(v[0], default=str)[:400]}")
                else:
                    print(f"  .{k}: list len={len(v)} sample={v[:3]!r:.200}")
            elif isinstance(v, dict):
                print(f"  .{k}: dict keys={sorted(v.keys())[:15]}")
            else:
                s = repr(v)
                if len(s) > 200:
                    s = s[:200] + "..."
                print(f"  .{k}: {type(v).__name__} = {s}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/locomo-leaderboard-checkpoint.json")
