"""Diagnostic: confirm the proof_count window-bias hypothesis for Phase A.

Hypothesis (from RASPUTIN_SESSION_STATE.md §7):
  proof_norm is a token-overlap proxy. Adjacent w5s2 windows share ~60% tokens.
  So for any given question, the ~45 near-duplicate windows all co-boost each other
  (high proof_norm), while the discriminating fact has unique tokens and is demoted
  (low proof_norm). This systematically pushes the gold-bearing fact down the rank.

Why this is Path A (not Path B):
  The checkpoint persists a ``chunks`` list of 60 formatted strings per question --
  the top-60 sent to the answer model after multiplicative rerank. The chunk type
  can be reconstructed from the prefix:
    - starts with "[Inference]"               -> chunk_type = inference fact
    - starts with "[HH:MM am/pm on DD Mon, YYYY]" -> chunk_type = window
    - anything else (with " | Involving:" etc.) -> chunk_type = world/experience fact
  We recompute proof_norm using the exact formula from cross_encoder.py:
    n = 1 + count(other candidates sharing >= 3 tokens of len >= 4)
    proof_norm = clamp(0.5 + log(n)/10, [0, 1])

What this measures:
  This is the proof_norm distribution of the *final pool that reached the answer
  model* (post-rerank), not the pre-rerank pool of ~70 candidates. If the hypothesis
  is true and the window-bias pushed the gold fact out of the top-60 for many
  questions, we'll see (a) windows dominating the top-60 with high proof_norm and
  (b) facts that did make it clustered at low proof_norm. This is still strong
  evidence of structural window-bias because the pool BEFORE rerank was even more
  window-heavy (45 windows + 25 facts + 10 bm25 per the current defaults) and the
  proof_boost would have amplified that further.

Threshold per RASPUTIN_NEXT_ACTION.md §Step 1:
  window mean - fact mean >= 0.1  -> hypothesis CONFIRMED, park Phase A
  window mean - fact mean <  0.05 -> hypothesis NOT confirmed, dig deeper

Output sections:
  A. Per-question chunk-type composition of top-60
  B. Proof_norm distribution (mean/median/stdev/P90) per chunk_type
  C. Correct vs incorrect: does window dominance correlate with wrong answers?
  D. Sample pathological cases (questions where the gold fact was probably displaced)
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CHECKPOINT = Path("benchmarks/results/locomo-leaderboard-checkpoint.json")

TIME_RE = re.compile(r"^\[(\d{1,2}:\d{2} (?:am|pm) on \d+ \w+, \d{4})\]")
INFERENCE_PREFIX = "[Inference]"


def classify_chunk(chunk: str) -> str:
    """Return one of: 'window', 'inference', 'fact'."""
    s = chunk.lstrip()
    if s.startswith(INFERENCE_PREFIX):
        return "inference"
    if TIME_RE.match(s):
        return "window"
    return "fact"  # world/experience facts (no tag prefix)


def compute_proof_norm_for_pool(chunks: list[str]) -> list[float]:
    """Exact port of cross_encoder._compute_proof_norm over a list of chunk strings."""
    token_sets: list[set[str]] = []
    for ch in chunks:
        text = ch.lower()
        toks = {w for w in re.findall(r"\w+", text) if len(w) >= 4}
        token_sets.append(toks)

    out: list[float] = []
    for i, ti in enumerate(token_sets):
        if not ti:
            out.append(0.5)
            continue
        supports = 0
        for j, tj in enumerate(token_sets):
            if i == j:
                continue
            if len(ti & tj) >= 3:
                supports += 1
        n = supports + 1  # self
        out.append(min(1.0, max(0.0, 0.5 + math.log(n) / 10.0)))
    return out


def pct(p: float, arr: list[float]) -> float:
    """Percentile without numpy."""
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def normalize(s: str) -> set[str]:
    return set(w for w in re.findall(r"\w+", s.lower()) if len(w) >= 4)


def main() -> None:
    if not CHECKPOINT.exists():
        sys.exit(f"Checkpoint not found at {CHECKPOINT}")

    print(f"Loading {CHECKPOINT} ...")
    ckpt = json.loads(CHECKPOINT.read_text())
    results = ckpt.get("results", [])
    print(f"Total result entries: {len(results)}\n")

    # Per-chunk-type proof_norm values (across all questions)
    pn_by_type: dict[str, list[float]] = defaultdict(list)

    # Per-question stats
    n_chunks_by_type_per_q: list[Counter] = []
    top10_type_share: Counter = Counter()  # in top-10 by list order (reranker rank)
    top5_type_share: Counter = Counter()

    # Correlation: does top-10 window share vary between correct and incorrect?
    correct_top10_window_share: list[float] = []
    incorrect_top10_window_share: list[float] = []

    # Proof_norm specifically for the chunk that contains the gold answer
    gold_chunk_proof_norms_by_type: dict[str, list[float]] = defaultdict(list)
    # Rank of the gold-bearing chunk in the top-60 (0 = rank 1)
    gold_chunk_rank_by_type: dict[str, list[int]] = defaultdict(list)

    # Pathological sample cases
    pathological: list[dict] = []

    for r in results:
        chunks = r.get("chunks") or []
        if not chunks:
            continue
        types = [classify_chunk(c) for c in chunks]
        proof = compute_proof_norm_for_pool(chunks)

        type_counter = Counter(types)
        n_chunks_by_type_per_q.append(type_counter)

        for i, (t, p) in enumerate(zip(types, proof)):
            pn_by_type[t].append(p)

        # Top-10 / Top-5 composition
        for t in types[:10]:
            top10_type_share[t] += 1
        for t in types[:5]:
            top5_type_share[t] += 1

        # Does top-10 window share differ for correct vs incorrect?
        if types:
            top10_windows = sum(1 for t in types[:10] if t == "window")
            share = top10_windows / min(len(types), 10)
            if r.get("correct"):
                correct_top10_window_share.append(share)
            else:
                incorrect_top10_window_share.append(share)

        # Gold-chunk analysis: find the chunk that best matches the gold answer by token overlap
        gold = (r.get("gold") or "").strip()
        if gold:
            gold_toks = normalize(gold)
            if gold_toks:
                best_rank = -1
                best_overlap = 0
                best_type = None
                best_pn = None
                for idx, (ch, t, p) in enumerate(zip(chunks, types, proof)):
                    ov = len(gold_toks & normalize(ch))
                    if ov > best_overlap:
                        best_overlap = ov
                        best_rank = idx
                        best_type = t
                        best_pn = p
                # Only count as "gold-bearing chunk found" if substantial overlap
                if best_overlap >= max(2, len(gold_toks) // 2) and best_type is not None:
                    gold_chunk_proof_norms_by_type[best_type].append(best_pn)  # type: ignore[arg-type]
                    gold_chunk_rank_by_type[best_type].append(best_rank)

        # Pathological flagging: incorrect answer, top-10 is 100% windows, a fact exists lower
        if not r.get("correct") and types[:10].count("window") >= 9:
            fact_positions = [i for i, t in enumerate(types) if t in ("fact", "inference")]
            fact_pns = [proof[i] for i in fact_positions]
            window_pns_top10 = [proof[i] for i, t in enumerate(types[:10]) if t == "window"]
            if fact_positions and window_pns_top10:
                pathological.append({
                    "conv_id": r.get("conv_id"),
                    "qi": r.get("qi"),
                    "question": r.get("question"),
                    "gold": gold[:80],
                    "predicted": (r.get("predicted") or "")[:80],
                    "top10_window_share": types[:10].count("window") / 10,
                    "mean_window_pn_top10": statistics.mean(window_pns_top10),
                    "mean_fact_pn": statistics.mean(fact_pns) if fact_pns else 0,
                    "best_fact_rank": min(fact_positions) if fact_positions else -1,
                })

    # ------------------------- REPORT -------------------------

    print("=" * 70)
    print("A. Per-question chunk-type composition of top-60 (post-rerank)")
    print("=" * 70)
    if n_chunks_by_type_per_q:
        shares_per_type: dict[str, list[float]] = defaultdict(list)
        for ct in n_chunks_by_type_per_q:
            total = sum(ct.values()) or 1
            for t in ("window", "fact", "inference"):
                shares_per_type[t].append(ct.get(t, 0) / total)
        for t in ("window", "fact", "inference"):
            arr = shares_per_type[t]
            print(
                f"  {t:10s} share mean={statistics.mean(arr):.3f}  "
                f"median={statistics.median(arr):.3f}  "
                f"P10={pct(0.10, arr):.3f}  P90={pct(0.90, arr):.3f}"
            )

    total_chunks = sum(sum(ct.values()) for ct in n_chunks_by_type_per_q) or 1
    print(f"\n  Grand-total chunk counts across {len(n_chunks_by_type_per_q)} questions:")
    for t, cnt in sorted(
        Counter({t: sum(ct.get(t, 0) for ct in n_chunks_by_type_per_q) for t in ("window", "fact", "inference")}).items(),
        key=lambda kv: -kv[1],
    ):
        print(f"    {t:10s} {cnt:>7d}  ({100*cnt/total_chunks:.1f}%)")

    print("\n  Top-10 rank composition (which type dominates near the top):")
    t10 = sum(top10_type_share.values()) or 1
    for t, cnt in top10_type_share.most_common():
        print(f"    {t:10s} {cnt:>7d}  ({100*cnt/t10:.1f}% of top-10 slots)")

    print("\n  Top-5 rank composition:")
    t5 = sum(top5_type_share.values()) or 1
    for t, cnt in top5_type_share.most_common():
        print(f"    {t:10s} {cnt:>7d}  ({100*cnt/t5:.1f}% of top-5 slots)")

    print()
    print("=" * 70)
    print("B. Proof_norm distribution per chunk_type (THE KEY NUMBER)")
    print("=" * 70)
    print(f"  {'type':10s}  {'N':>8s}  {'mean':>7s}  {'median':>7s}  {'stdev':>7s}  {'P10':>7s}  {'P90':>7s}")
    means: dict[str, float] = {}
    for t in ("window", "fact", "inference"):
        arr = pn_by_type.get(t, [])
        if not arr:
            continue
        m = statistics.mean(arr)
        means[t] = m
        print(
            f"  {t:10s}  {len(arr):>8d}  {m:>7.4f}  {statistics.median(arr):>7.4f}  "
            f"{statistics.stdev(arr) if len(arr) > 1 else 0:>7.4f}  "
            f"{pct(0.10, arr):>7.4f}  {pct(0.90, arr):>7.4f}"
        )

    gap_wf = means.get("window", 0) - means.get("fact", 0)
    gap_wi = means.get("window", 0) - means.get("inference", 0)
    print(f"\n  window_mean - fact_mean      = {gap_wf:+.4f}")
    print(f"  window_mean - inference_mean = {gap_wi:+.4f}")

    print("\n  THRESHOLD (per RASPUTIN_NEXT_ACTION.md):")
    if gap_wf >= 0.10:
        verdict = "CONFIRMED (gap >= 0.10) -- proof_count is window-biased on LoCoMo"
    elif gap_wf < 0.05:
        verdict = "NOT CONFIRMED (gap < 0.05) -- look elsewhere for the Phase A regression"
    else:
        verdict = f"INCONCLUSIVE (0.05 <= gap {gap_wf:.4f} < 0.10) -- inspect pathological cases"
    print(f"    -> {verdict}")

    print()
    print("=" * 70)
    print("C. Does top-10 window dominance correlate with answer correctness?")
    print("=" * 70)
    if correct_top10_window_share and incorrect_top10_window_share:
        cmean = statistics.mean(correct_top10_window_share)
        imean = statistics.mean(incorrect_top10_window_share)
        print(f"  Correct   answers: mean top-10 window share = {cmean:.3f}  (N={len(correct_top10_window_share)})")
        print(f"  Incorrect answers: mean top-10 window share = {imean:.3f}  (N={len(incorrect_top10_window_share)})")
        print(f"  Difference (incorrect - correct)            = {imean - cmean:+.4f}")
        print("  Interpretation: if window share is HIGHER on incorrect answers,")
        print("  window-bias is pushing discriminating facts out of the top-10.")

    print()
    print("=" * 70)
    print("D. Gold-bearing-chunk analysis")
    print("   (chunk with highest token overlap vs the gold answer, if overlap >= max(2, |gold|/2))")
    print("=" * 70)
    for t in ("window", "fact", "inference"):
        pns = gold_chunk_proof_norms_by_type.get(t, [])
        ranks = gold_chunk_rank_by_type.get(t, [])
        if not pns:
            print(f"  {t:10s} N=0 (no gold-bearing chunk classified as this type)")
            continue
        print(
            f"  {t:10s}  N={len(pns):>5d}  "
            f"proof_norm mean={statistics.mean(pns):.4f}  median={statistics.median(pns):.4f}  "
            f"rank mean={statistics.mean(ranks):.2f}  median={int(statistics.median(ranks))}"
        )

    print()
    print("=" * 70)
    print("E. Pathological samples (incorrect, top-10 nearly all windows, fact exists lower)")
    print(f"   Total flagged: {len(pathological)} / {len(results)}  ({100*len(pathological)/max(len(results),1):.1f}%)")
    print("=" * 70)
    # Show 5 with the largest proof_norm gap between top-10-windows and existing facts
    pathological.sort(key=lambda p: p["mean_window_pn_top10"] - p["mean_fact_pn"], reverse=True)
    for p in pathological[:5]:
        print(f"  conv={p['conv_id']} qi={p['qi']}")
        print(f"    Q:    {p['question'][:110]}")
        print(f"    gold: {p['gold']}")
        print(f"    pred: {p['predicted']}")
        print(
            f"    top10 window share = {p['top10_window_share']:.2f}  "
            f"mean window proof_norm (top10) = {p['mean_window_pn_top10']:.3f}  "
            f"mean fact proof_norm = {p['mean_fact_pn']:.3f}  "
            f"best fact rank in top-60 = {p['best_fact_rank']}"
        )
        print()


if __name__ == "__main__":
    main()
