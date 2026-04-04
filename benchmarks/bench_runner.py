#!/usr/bin/env python3
"""
RASPUTIN Memory — Benchmark Runner

Two benchmark modes:
  --mode production (default): Haiku answers, neutral judge, 60 chunks.
    Tracks retrieval quality over time. Results in {hash}-{bench}-production.json.
  --mode compare: gpt-4o-mini answers, generous judge, 60 chunks.
    Competition-comparable. Results in {hash}-{bench}-compare.json.

Three execution styles:
  Sequential: runs benchmark end-to-end in one process.
  Batch:      --submit / --retrieve / --finalize (Anthropic + OpenAI batch APIs, 50% cheaper).
  Compare:    --compare-to HASH (diff two commits).

Usage:
    python3 benchmarks/bench_runner.py locomo [--mode production|compare]
    python3 benchmarks/bench_runner.py locomo --submit [--mode production]
    python3 benchmarks/bench_runner.py locomo --retrieve
    python3 benchmarks/bench_runner.py locomo --finalize
    python3 benchmarks/bench_runner.py locomo --compare-to HASH
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "benchmarks" / "results"
HISTORY_FILE = RESULTS_DIR / "history.csv"

BENCH_SCRIPTS = {
    "locomo": "benchmarks/locomo_leaderboard_bench.py",
    "longmemeval": "benchmarks/longmemeval_bench.py",
    "frames": "benchmarks/frames_bench.py",
    "locomo-plus": "benchmarks/locomo_plus_bench.py",
}

HISTORY_FIELDS = [
    "commit",
    "commit_short",
    "date",
    "benchmark",
    "mode",
    "headline_score",
    "answer_model",
    "judge_model",
    "context_chunks",
    "search_limit",
    "note",
]

JUDGE_MODEL_PINNED = "gpt-4o-mini-2024-07-18"

JUDGE_PROMPT_GENEROUS = (
    "Is the system's answer correct? Be generous \u2014 if the answer captures the essential "
    "information from the ground truth, even if phrased differently or includes extra correct "
    "details, score it as CORRECT. Only score WRONG if the answer is factually incorrect, "
    "missing the key information, or says it doesn't know when the answer was available."
)

JUDGE_PROMPT_NEUTRAL = (
    "Is the system's answer correct? Score CORRECT only if the answer contains the specific "
    "information asked for. Score WRONG if the answer is vague, missing key facts, or incorrect. "
    "Do not give credit for answers that are technically true but don't answer the question."
)

MODE_DEFAULTS = {
    "production": {
        "answer_model": "claude-haiku-4-5-20251001",
        "judge_model": JUDGE_MODEL_PINNED,
        "judge_prompt": JUDGE_PROMPT_NEUTRAL,
        "context_chunks": 60,
        "search_limit": 60,
    },
    "compare": {
        "answer_model": "gpt-4o-mini",
        "judge_model": JUDGE_MODEL_PINNED,
        "judge_prompt": JUDGE_PROMPT_GENEROUS,
        "context_chunks": 60,
        "search_limit": 60,
    },
}


def get_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    return result.stdout.strip()


def repo_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    return bool(result.stdout.strip())


def result_filename(commit: str, benchmark: str, mode: str) -> str:
    return f"{commit}-{benchmark}-{mode}.json"


def state_path(commit: str, benchmark: str) -> Path:
    return RESULTS_DIR / f"{commit}-{benchmark}-state.json"


def load_state(commit: str, benchmark: str) -> dict[str, Any]:
    p = state_path(commit, benchmark)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"commit": commit, "benchmark": benchmark, "phase": "init"}


def save_state(st: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = state_path(st["commit"], st["benchmark"])
    with open(p, "w") as f:
        json.dump(st, f, indent=2)


def get_mode_config(mode: str) -> dict[str, Any]:
    cfg = MODE_DEFAULTS[mode].copy()
    cfg["answer_model"] = os.environ.get("BENCH_ANSWER_MODEL", cfg["answer_model"])
    cfg["mode"] = mode
    return cfg


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE) as f:
        return list(csv.DictReader(f))


def save_history_row(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_history()
    existing_keys = {r["commit"] + r["benchmark"] + r.get("mode", "") for r in existing}
    key = row["commit"] + row["benchmark"] + row.get("mode", "")
    if key in existing_keys:
        print(f"  History already has entry for {row['commit_short']} / {row['benchmark']} / {row.get('mode', '')}")
        return

    write_header = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_result_file(commit: str, benchmark: str, mode: str = "production") -> dict | None:
    for m in [mode, "production", "compare", ""]:
        fname = f"{commit}-{benchmark}-{m}.json" if m else f"{commit}-{benchmark}.json"
        p = RESULTS_DIR / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def compare_results(old: dict, new: dict) -> list[dict]:
    deltas = []
    old_scores = old.get("scores", {})
    new_scores = new.get("scores", {})
    if not old_scores:
        old_scores = {k: v for k, v in old.items() if isinstance(v, (int, float)) and k != "total"}
    if not new_scores:
        new_scores = {k: v for k, v in new.items() if isinstance(v, (int, float)) and k != "total"}
    for key in sorted(set(old_scores) | set(new_scores)):
        ov, nv = old_scores.get(key), new_scores.get(key)
        if ov is not None and nv is not None and isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            delta = nv - ov
            deltas.append({"metric": key, "old": ov, "new": nv, "delta": delta, "improved": delta > 0})
    return deltas


def print_comparison(deltas: list[dict], old_commit: str, new_commit: str) -> bool:
    print(f"\n  Comparison: {old_commit[:8]} \u2192 {new_commit[:8]}")
    print(f"  {'Metric':<30s} {'Old':>8s} {'New':>8s} {'Delta':>10s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 10}")
    has_regression = False
    for d in deltas:
        marker = "+" if d["improved"] else "-"
        symbol = "\u2713" if d["improved"] else "\u2717"
        print(f"  {d['metric']:<30s} {d['old']:>8.3f} {d['new']:>8.3f} {marker}{abs(d['delta']):>8.3f} {symbol}")
        if not d["improved"] and abs(d["delta"]) > 0.001:
            has_regression = True
    return has_regression


def offer_revert(deltas: list[dict]) -> None:
    regressions = [d for d in deltas if not d["improved"] and abs(d["delta"]) > 0.001]
    if not regressions:
        return
    print(f"\n  REGRESSION in: {', '.join(d['metric'] for d in regressions)}")
    try:
        answer = input("  Revert HEAD? (y/n): ").strip().lower()
    except EOFError:
        return
    if answer == "y":
        subprocess.run(["git", "revert", "HEAD", "--no-edit"], cwd=str(REPO))
        print("  Reverted.")


def set_bench_env(env: dict, mode_cfg: dict) -> None:
    env["BENCH_ANSWER_MODEL"] = mode_cfg["answer_model"]
    env["BENCH_JUDGE_MODEL"] = mode_cfg["judge_model"]
    env["BENCH_CONTEXT_CHUNKS"] = str(mode_cfg["context_chunks"])
    env["BENCH_SEARCH_LIMIT"] = str(mode_cfg["search_limit"])
    env["BENCH_JUDGE_PROMPT"] = mode_cfg["judge_prompt"]
    env["BENCH_MODE"] = mode_cfg["mode"]


# ─── Batch: --submit ─────────────────────────────────────────


def batch_submit(benchmark: str, commit: str, mode_cfg: dict, extra_args: list[str]) -> None:
    script = BENCH_SCRIPTS[benchmark]
    search_output = RESULTS_DIR / f"{commit}-{benchmark}-search.json"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BENCH_COMMIT"] = commit
    env["BENCH_SEARCH_OUTPUT"] = str(search_output)
    set_bench_env(env, mode_cfg)

    cmd = [sys.executable, "-u", str(REPO / script), "--reset", "--search-only"] + extra_args
    print(f"  Phase 1 (search): {script}")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO))
    if proc.returncode != 0:
        print(f"  Search failed (exit {proc.returncode})")
        sys.exit(1)
    if not search_output.exists():
        print(f"  No search output at {search_output}")
        sys.exit(1)

    from benchmarks.batch_api import anthropic_create_batch

    with open(search_output) as f:
        search_data = json.load(f)

    answer_model = mode_cfg["answer_model"]
    max_chunks = mode_cfg["context_chunks"]

    requests = []
    for item in search_data.get("items", []):
        chunks = item.get("chunks", [])
        context = "\n".join(f"- {c}" for c in chunks[:max_chunks])
        prompt = (
            "You are answering questions about a conversation based on retrieved memory snippets.\n"
            "Answer concisely in 1-3 sentences. Be direct and specific.\n"
            "If NO relevant facts exist in the memories, say "
            '"I don\'t have enough information to answer this question."\n\n'
            f"Memories:\n{context}\n\nQuestion: {item['question']}\nAnswer:"
        )
        requests.append(
            {
                "custom_id": item["id"],
                "params": {
                    "model": answer_model,
                    "max_tokens": 150,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    print(f"  Submitting {len(requests)} answer requests ({answer_model})")
    batch = anthropic_create_batch(requests)
    print(f"  Batch: {batch['id']} ({batch['processing_status']})")

    st = load_state(commit, benchmark)
    st.update(
        {
            "phase": "answers-submitted",
            "mode": mode_cfg["mode"],
            "search_output": str(search_output),
            "answer_batch_id": batch["id"],
            "answer_model": answer_model,
            "total_questions": len(requests),
        }
    )
    save_state(st)
    print("  Next: --retrieve")


# ─── Batch: --retrieve ───────────────────────────────────────


def batch_retrieve(benchmark: str, commit: str, mode_cfg: dict) -> None:
    from benchmarks.batch_api import (
        anthropic_get_results,
        anthropic_poll_batch,
        openai_create_batch,
        openai_upload_jsonl,
    )

    st = load_state(commit, benchmark)
    if st.get("phase") != "answers-submitted":
        print(f"  State phase is '{st.get('phase')}', expected 'answers-submitted'.")
        sys.exit(1)

    batch_id = st["answer_batch_id"]
    print(f"  Polling Anthropic batch {batch_id}...")
    anthropic_poll_batch(batch_id)

    print("  Downloading answers...")
    results = anthropic_get_results(batch_id)

    answers: dict[str, str] = {}
    for r in results:
        cid = r.get("custom_id", "")
        rd = r.get("result", {})
        if rd.get("type") == "succeeded":
            content = rd.get("message", {}).get("content", [{}])
            answers[cid] = content[0].get("text", "") if content else ""
        else:
            answers[cid] = ""

    answers_file = RESULTS_DIR / f"{commit}-{benchmark}-answers.json"
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent=2)
    print(f"  {sum(1 for v in answers.values() if v)}/{len(answers)} answers retrieved")

    with open(st["search_output"]) as f:
        search_data = json.load(f)
    items_by_id = {item["id"]: item for item in search_data.get("items", [])}

    judge_prompt = mode_cfg["judge_prompt"]
    judge_model = mode_cfg["judge_model"]

    lines = []
    for qid, answer in answers.items():
        item = items_by_id.get(qid, {})
        prompt = (
            "You are evaluating an AI memory system's answer.\n\n"
            f"Question: {item.get('question', '')}\n"
            f"Ground Truth Answer: {item.get('gold', '')}\n"
            f"System Answer: {answer}\n\n"
            f"{judge_prompt}\n\n"
            "Reply with exactly one word: CORRECT or WRONG"
        )
        lines.append(
            json.dumps(
                {
                    "custom_id": qid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": judge_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 10,
                    },
                }
            )
        )

    print(f"  Uploading {len(lines)} judge requests ({judge_model})...")
    file_id = openai_upload_jsonl("\n".join(lines) + "\n")
    batch = openai_create_batch(file_id, metadata={"description": f"judge-{benchmark}-{commit[:8]}"})
    print(f"  Judge batch: {batch['id']}")

    st.update(
        {
            "phase": "judge-submitted",
            "answers_file": str(answers_file),
            "judge_batch_id": batch["id"],
            "judge_file_id": file_id,
        }
    )
    save_state(st)
    print("  Next: --finalize")


# ─── Batch: --finalize ───────────────────────────────────────


def batch_finalize(benchmark: str, commit: str, mode_cfg: dict) -> None:
    from benchmarks.batch_api import openai_get_results, openai_poll_batch

    st = load_state(commit, benchmark)
    if st.get("phase") != "judge-submitted":
        print(f"  State phase is '{st.get('phase')}', expected 'judge-submitted'.")
        sys.exit(1)

    batch_id = st["judge_batch_id"]
    print(f"  Polling OpenAI batch {batch_id}...")
    batch = openai_poll_batch(batch_id)

    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        print("  No output file. Batch may have failed.")
        sys.exit(1)

    print("  Downloading judge results...")
    results = openai_get_results(output_file_id)

    scores: dict[str, float] = {}
    for r in results:
        cid = r.get("custom_id", "")
        if r.get("error"):
            scores[cid] = 0.0
        elif r.get("response", {}).get("status_code") == 200:
            text = r["response"]["body"].get("choices", [{}])[0].get("message", {}).get("content", "")
            scores[cid] = 1.0 if "CORRECT" in text.upper() else 0.0
        else:
            scores[cid] = 0.0

    total = len(scores)
    correct = sum(1 for v in scores.values() if v >= 1.0)
    accuracy = correct / total * 100 if total else 0

    mode = mode_cfg["mode"]
    final_result = {
        "commit": commit,
        "commit_short": commit[:8],
        "benchmark": benchmark,
        "mode": mode,
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "methodology": {
            "answer_model": mode_cfg["answer_model"],
            "judge_model": mode_cfg["judge_model"],
            "context_chunks": mode_cfg["context_chunks"],
            "search_limit": mode_cfg["search_limit"],
            "judge_prompt_style": "generous" if mode == "compare" else "neutral",
        },
        "scores": scores,
    }

    rf = RESULTS_DIR / result_filename(commit, benchmark, mode)
    with open(rf, "w") as f:
        json.dump(final_result, f, indent=2)

    st["phase"] = "complete"
    st["accuracy"] = accuracy
    save_state(st)

    print(f"\n  RESULT ({mode} mode): {accuracy:.2f}% ({correct}/{total})")
    print(f"  Saved: {rf}")

    save_history_row(
        {
            "commit": commit,
            "commit_short": commit[:8],
            "date": datetime.now().isoformat(),
            "benchmark": benchmark,
            "mode": mode,
            "headline_score": f"{accuracy:.2f}",
            "answer_model": mode_cfg["answer_model"],
            "judge_model": mode_cfg["judge_model"],
            "context_chunks": str(mode_cfg["context_chunks"]),
            "search_limit": str(mode_cfg["search_limit"]),
            "note": "batch",
        }
    )


# ─── Sequential (legacy) ─────────────────────────────────────


def run_sequential(benchmark: str, commit: str, mode_cfg: dict, extra_args: list[str]) -> None:
    script = BENCH_SCRIPTS.get(benchmark)
    if not script:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)

    mode = mode_cfg["mode"]
    rf = RESULTS_DIR / result_filename(commit, benchmark, mode)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BENCH_RESULT_FILE"] = str(rf)
    env["BENCH_COMMIT"] = commit
    set_bench_env(env, mode_cfg)

    cmd = [sys.executable, "-u", str(REPO / script), "--reset"] + extra_args
    print(f"  Running: {script} {' '.join(extra_args)}")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO))

    if proc.returncode != 0:
        print(f"  Benchmark failed (exit {proc.returncode})")
        sys.exit(1)

    result = None
    if rf.exists():
        with open(rf) as f:
            result = json.load(f)
    else:
        standard_files = {
            "locomo": RESULTS_DIR / "locomo-leaderboard-v1.json",
            "longmemeval": RESULTS_DIR / "longmemeval-results.json",
            "frames": RESULTS_DIR / "frames-results.json",
            "locomo-plus": RESULTS_DIR / "locomo-plus-results.json",
        }
        fallback = standard_files.get(benchmark)
        if fallback and fallback.exists():
            with open(fallback) as f:
                result = json.load(f)

    if not result:
        print("\n  No result file produced.")
        sys.exit(1)

    result["commit"] = commit
    result["commit_short"] = commit[:8]
    result["mode"] = mode
    result["methodology"] = {
        "answer_model": mode_cfg["answer_model"],
        "judge_model": mode_cfg["judge_model"],
        "context_chunks": mode_cfg["context_chunks"],
        "search_limit": mode_cfg["search_limit"],
        "judge_prompt_style": "generous" if mode == "compare" else "neutral",
    }

    with open(rf, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result ({mode}): {rf}")

    headline = result.get("headline_score") or result.get("overall_accuracy") or result.get("overall")
    save_history_row(
        {
            "commit": commit,
            "commit_short": commit[:8],
            "date": datetime.now().isoformat(),
            "benchmark": benchmark,
            "mode": mode,
            "headline_score": f"{headline:.2f}" if isinstance(headline, (int, float)) else str(headline or ""),
            "answer_model": mode_cfg["answer_model"],
            "judge_model": mode_cfg["judge_model"],
            "context_chunks": str(mode_cfg["context_chunks"]),
            "search_limit": str(mode_cfg["search_limit"]),
            "note": "",
        }
    )


# ─── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RASPUTIN benchmark runner")
    parser.add_argument("benchmark", nargs="?", choices=list(BENCH_SCRIPTS.keys()))
    parser.add_argument("--mode", choices=["production", "compare"], default="production")
    parser.add_argument("--submit", action="store_true", help="Batch: search + submit answer batch")
    parser.add_argument("--retrieve", action="store_true", help="Batch: retrieve answers + submit judge batch")
    parser.add_argument("--finalize", action="store_true", help="Batch: retrieve judge + score")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compare-to", type=str, default=None)
    parser.add_argument("--answer-model", type=str, default=None, help="Override answer model")
    parser.add_argument("--constraints", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conversations", type=str, default=None)
    args = parser.parse_args()

    if args.compare_to and args.benchmark:
        commit = get_commit_hash()
        old_result = load_result_file(args.compare_to, args.benchmark, args.mode)
        new_result = load_result_file(commit, args.benchmark, args.mode)
        if not old_result:
            print(f"No result for {args.compare_to[:8]} / {args.benchmark}")
            sys.exit(1)
        if not new_result:
            print(f"No result for {commit[:8]} / {args.benchmark}. Run benchmark first.")
            sys.exit(1)
        deltas = compare_results(old_result, new_result)
        if deltas:
            if print_comparison(deltas, args.compare_to, commit):
                offer_revert(deltas)
        else:
            print("  No comparable metrics found.")
        sys.exit(0)

    if not args.benchmark:
        parser.print_help()
        sys.exit(1)

    mode_cfg = get_mode_config(args.mode)
    if args.answer_model:
        mode_cfg["answer_model"] = args.answer_model

    commit = get_commit_hash()

    extra_args: list[str] = []
    if args.limit:
        extra_args.extend(["--limit", str(args.limit)])
    if args.conversations is not None:
        extra_args.extend(["--conversations", args.conversations])
    if args.constraints and args.benchmark == "locomo-plus":
        extra_args.append("--constraints")

    print(f"Benchmark: {args.benchmark}")
    print(f"Mode:      {args.mode}")
    print(f"Commit:    {commit[:8]}")
    print(f"Answer:    {mode_cfg['answer_model']}")
    print(f"Judge:     {mode_cfg['judge_model']} ({'generous' if args.mode == 'compare' else 'neutral'})")
    print(f"Chunks:    {mode_cfg['context_chunks']}")
    print()

    if args.submit:
        if repo_is_dirty():
            print("ERROR: uncommitted changes. Commit first.")
            sys.exit(1)
        batch_submit(args.benchmark, commit, mode_cfg, extra_args)
    elif args.retrieve:
        batch_retrieve(args.benchmark, commit, mode_cfg)
    elif args.finalize:
        batch_finalize(args.benchmark, commit, mode_cfg)
    else:
        if repo_is_dirty():
            print("ERROR: uncommitted changes. Commit first.")
            sys.exit(1)
        mode = args.mode
        rf = RESULTS_DIR / result_filename(commit, args.benchmark, mode)
        if rf.exists() and not args.force:
            print(f"Already benchmarked {commit[:8]} / {args.benchmark} / {mode}. Use --force.")
            sys.exit(0)
        run_sequential(args.benchmark, commit, mode_cfg, extra_args)


if __name__ == "__main__":
    main()
