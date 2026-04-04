#!/usr/bin/env python3
"""
RASPUTIN Memory — Benchmark Runner with Git Integrity

Enforces: clean repo, commit-hash keying, history tracking, regression detection.

Usage:
    python3 benchmarks/bench_runner.py locomo [--force] [--limit N]
    python3 benchmarks/bench_runner.py longmemeval [--force]
    python3 benchmarks/bench_runner.py frames [--force]
    python3 benchmarks/bench_runner.py locomo-plus [--force] [--constraints]
    python3 benchmarks/bench_runner.py --compare-to HASH locomo
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
    "headline_score",
    "answer_model",
    "judge_model",
    "context_chunks",
    "search_limit",
    "note",
]


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


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE) as f:
        return list(csv.DictReader(f))


def save_history_row(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_history()
    existing_keys = {r["commit"] + r["benchmark"] for r in existing}
    key = row["commit"] + row["benchmark"]
    if key in existing_keys:
        print(f"  History already has entry for {row['commit_short']} / {row['benchmark']}, skipping.")
        return

    write_header = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_result_file(commit: str, benchmark: str) -> dict | None:
    result_file = RESULTS_DIR / f"{commit}-{benchmark}.json"
    if result_file.exists():
        with open(result_file) as f:
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

    all_keys = sorted(set(old_scores.keys()) | set(new_scores.keys()))
    for key in all_keys:
        ov = old_scores.get(key)
        nv = new_scores.get(key)
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
        print("\n  No regressions detected.")
        return

    print(f"\n  REGRESSION in: {', '.join(d['metric'] for d in regressions)}")
    try:
        answer = input("  Revert HEAD? (y/n): ").strip().lower()
    except EOFError:
        return

    if answer == "y":
        subprocess.run(["git", "revert", "HEAD", "--no-edit"], cwd=str(REPO))
        print("  Reverted. Regression data preserved in revert commit.")


def run_bench(benchmark: str, extra_args: list[str], commit: str) -> dict | None:
    script = BENCH_SCRIPTS.get(benchmark)
    if not script:
        print(f"Unknown benchmark: {benchmark}")
        return None

    result_file = RESULTS_DIR / f"{commit}-{benchmark}.json"

    cmd = [sys.executable, "-u", str(REPO / script), "--reset"] + extra_args
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BENCH_RESULT_FILE"] = str(result_file)
    env["BENCH_COMMIT"] = commit

    print(f"  Running: {script} {' '.join(extra_args)}")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO))

    if proc.returncode != 0:
        print(f"  Benchmark failed with exit code {proc.returncode}")
        return None

    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)

    standard_files = {
        "locomo": RESULTS_DIR / "locomo-leaderboard-v1.json",
        "longmemeval": RESULTS_DIR / "longmemeval-results.json",
        "frames": RESULTS_DIR / "frames-results.json",
        "locomo-plus": RESULTS_DIR / "locomo-plus-results.json",
    }
    fallback = standard_files.get(benchmark)
    if fallback and fallback.exists():
        with open(fallback) as f:
            data = json.load(f)
        data["commit"] = commit
        data["commit_short"] = commit[:8]
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        return data
    return None


def main():
    parser = argparse.ArgumentParser(description="RASPUTIN benchmark runner with git integrity")
    parser.add_argument("benchmark", nargs="?", choices=list(BENCH_SCRIPTS.keys()), help="Benchmark to run")
    parser.add_argument("--force", action="store_true", help="Re-run even if commit already benchmarked")
    parser.add_argument("--compare-to", type=str, default=None, help="Compare current result to a previous commit hash")
    parser.add_argument("--constraints", action="store_true", help="Enable constraints (locomo-plus only)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions/conversations")
    parser.add_argument("--conversations", type=str, default=None, help="Conversation indices (locomo)")
    args = parser.parse_args()

    if args.compare_to and args.benchmark:
        commit = get_commit_hash()
        old_result = load_result_file(args.compare_to, args.benchmark)
        new_result = load_result_file(commit, args.benchmark)
        if not old_result:
            print(f"No result file for {args.compare_to[:8]} / {args.benchmark}")
            sys.exit(1)
        if not new_result:
            print(f"No result file for {commit[:8]} / {args.benchmark}. Run the benchmark first.")
            sys.exit(1)
        deltas = compare_results(old_result, new_result)
        if deltas:
            has_regression = print_comparison(deltas, args.compare_to, commit)
            if has_regression:
                offer_revert(deltas)
        else:
            print("  No comparable numeric metrics found.")
        sys.exit(0)

    if not args.benchmark:
        parser.print_help()
        sys.exit(1)

    if repo_is_dirty():
        print("ERROR: uncommitted changes. Commit first \u2014 benchmark must be tied to a hash.")
        sys.exit(1)

    commit = get_commit_hash()
    existing = load_result_file(commit, args.benchmark)
    if existing and not args.force:
        print(f"Already benchmarked commit {commit[:8]} / {args.benchmark}. Use --force to re-run.")
        sys.exit(0)

    extra_args = []
    if args.limit:
        extra_args.extend(["--limit", str(args.limit)])
    if args.conversations is not None:
        extra_args.extend(["--conversations", args.conversations])
    if args.constraints and args.benchmark == "locomo-plus":
        extra_args.append("--constraints")

    print(f"Benchmark: {args.benchmark}")
    print(f"Commit:    {commit[:8]} ({commit})")
    print(f"Date:      {datetime.now().isoformat()}")
    print()

    result = run_bench(args.benchmark, extra_args, commit)
    if not result:
        print("\n  No result file produced.")
        sys.exit(1)

    answer_model = os.environ.get("BENCH_ANSWER_MODEL", "claude-haiku-4-5-20251001")
    judge_model = "gpt-4o-mini"
    context_chunks = int(os.environ.get("BENCH_CONTEXT_CHUNKS", "10"))
    search_limit = int(os.environ.get("BENCH_SEARCH_LIMIT", "10"))

    result["commit"] = commit
    result["commit_short"] = commit[:8]
    result["methodology"] = {
        "answer_model": answer_model,
        "judge_model": judge_model,
        "context_chunks": context_chunks,
        "search_limit": search_limit,
        "note": (
            "Leaderboard systems evaluated with their own answer models \u2014 "
            "direct score comparison not valid; retrieval quality comparison "
            "requires same answer model across all systems"
        ),
    }

    result_file = RESULTS_DIR / f"{commit}-{args.benchmark}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved: {result_file}")

    headline = result.get("headline_score") or result.get("overall_accuracy") or result.get("overall")
    save_history_row(
        {
            "commit": commit,
            "commit_short": commit[:8],
            "date": datetime.now().isoformat(),
            "benchmark": args.benchmark,
            "headline_score": f"{headline:.2f}" if isinstance(headline, (int, float)) else str(headline or ""),
            "answer_model": answer_model,
            "judge_model": judge_model,
            "context_chunks": str(context_chunks),
            "search_limit": str(search_limit),
            "note": "",
        }
    )
    print(f"  History updated: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
