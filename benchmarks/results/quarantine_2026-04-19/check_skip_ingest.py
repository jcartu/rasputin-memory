#!/usr/bin/env python3
"""Verify whether Phase A v1.json actually re-ran convs 26/30/41/42/43/44
or reused baseline answers for them.

Strategy: compare per-result content (question text, predicted_answer, score,
retrieved chunk IDs if present) between checkpoint.json and v1.json for the
'skipped' convs. If identical byte-for-byte, Phase A reused baseline outputs
for those convs -> -14.1pp comes ONLY from convs 47/48/49/50.
"""
from __future__ import annotations
import json
import hashlib
from collections import defaultdict

CHECKPOINT = "benchmarks/results/locomo-leaderboard-checkpoint.json"
V1 = "benchmarks/results/locomo-leaderboard-v1.json"

print("Loading checkpoint...")
with open(CHECKPOINT) as f:
    ckpt = json.load(f)
print("Loading v1...")
with open(V1) as f:
    v1 = json.load(f)

# structure probe
print(f"\ncheckpoint top-level keys: {list(ckpt.keys())[:10]}")
print(f"v1 top-level keys: {list(v1.keys())[:10]}")

# find the results array
def find_results(d):
    if isinstance(d, list):
        return d
    for k in ("results", "rows", "data", "predictions"):
        if k in d and isinstance(d[k], list):
            return d[k]
    return None

ckpt_rows = find_results(ckpt)
v1_rows = find_results(v1)
print(f"\ncheckpoint rows: {len(ckpt_rows) if ckpt_rows else 'N/A'}")
print(f"v1 rows: {len(v1_rows) if v1_rows else 'N/A'}")

if ckpt_rows and v1_rows:
    # sample row
    print(f"\nsample ckpt row keys: {list(ckpt_rows[0].keys())}")
    # group by conv_id
    def group(rows):
        g = defaultdict(list)
        for r in rows:
            cid = r.get("conversation_id") or r.get("conv_id") or r.get("sample_id", "?")
            g[cid].append(r)
        return g
    ckpt_g = group(ckpt_rows)
    v1_g = group(v1_rows)
    print(f"\nckpt convs: {sorted(ckpt_g.keys())[:20]}")
    print(f"v1 convs:   {sorted(v1_g.keys())[:20]}")

    # compare per-conv hash of predicted_answers
    def conv_hash(rows):
        # hash of sorted (question, predicted_answer) pairs
        pairs = []
        for r in rows:
            q = r.get("question", "")
            a = r.get("predicted_answer", r.get("answer", r.get("prediction", "")))
            pairs.append((q, a))
        pairs.sort()
        h = hashlib.md5()
        for q, a in pairs:
            h.update(q.encode())
            h.update(b"||")
            h.update(str(a).encode())
            h.update(b"\n")
        return h.hexdigest(), len(pairs)

    print("\nPer-conv comparison (ckpt vs v1 predicted_answer hash):")
    print(f"{'conv':<15} {'ckpt_n':>7} {'v1_n':>7}  {'ckpt_hash':>10}  {'v1_hash':>10}  same?")
    all_convs = sorted(set(ckpt_g.keys()) | set(v1_g.keys()))
    for cid in all_convs:
        ch, cn = conv_hash(ckpt_g.get(cid, []))
        vh, vn = conv_hash(v1_g.get(cid, []))
        same = "YES" if ch == vh else "NO"
        print(f"{str(cid):<15} {cn:>7} {vn:>7}  {ch[:10]}  {vh[:10]}  {same}")
