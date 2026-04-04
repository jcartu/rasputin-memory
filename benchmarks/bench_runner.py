#!/usr/bin/env python3
"""
RASPUTIN Memory — Benchmark Runner with Git Integrity + Batch API Support

Three execution modes:
  1. Legacy (sequential): runs benchmark end-to-end in one process
  2. Phased (batch): splits into search → submit-answers → retrieve → submit-judge → retrieve → score
  3. Compare: diff two commit results

Phase 1 requires clean repo (commit hash captured). Retrieval phases do not.

Usage:
    # Legacy mode (sequential, for quick tests)
    python3 benchmarks/bench_runner.py locomo [--force] [--limit N]

    # Phased mode (batch APIs, 50% cheaper)
    python3 benchmarks/bench_runner.py locomo --phase search
    python3 benchmarks/bench_runner.py locomo --phase submit-answers
    python3 benchmarks/bench_runner.py locomo --phase retrieve-answers
    python3 benchmarks/bench_runner.py locomo --phase submit-judge
    python3 benchmarks/bench_runner.py locomo --phase retrieve-judge

    # Compare
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
    "headline_score",
    "answer_model",
    "judge_model",
    "context_chunks",
    "search_limit",
    "note",
]

PHASES = ["search", "submit-answers", "retrieve-answers", "submit-judge", "retrieve-judge"]


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


def get_methodology() -> dict[str, Any]:
    return {
        "answer_model": os.environ.get("BENCH_ANSWER_MODEL", "claude-haiku-4-5-20251001"),
        "judge_model": "gpt-4o-mini",
        "context_chunks": int(os.environ.get("BENCH_CONTEXT_CHUNKS", "10")),
        "search_limit": int(os.environ.get("BENCH_SEARCH_LIMIT", "10")),
        "note": (
            "Leaderboard systems evaluated with their own answer models \u2014 "
            "direct score comparison not valid; retrieval quality comparison "
            "requires same answer model across all systems"
        ),
    }


# ─── Phase handlers ──────────────────────────────────────────


def phase_search(benchmark: str, commit: str, extra_args: list[str]) -> None:
    script = BENCH_SCRIPTS[benchmark]
    cmd = [sys.executable, "-u", str(REPO / script), "--reset", "--search-only"] + extra_args
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BENCH_COMMIT"] = commit

    search_output = RESULTS_DIR / f"{commit}-{benchmark}-search.json"
    env["BENCH_SEARCH_OUTPUT"] = str(search_output)

    print(f"  Phase 1 (search): {script}")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO))
    if proc.returncode != 0:
        print(f"  Search phase failed (exit {proc.returncode})")
        sys.exit(1)

    if not search_output.exists():
        print(f"  No search output at {search_output}")
        print("  Hint: benchmark script must support --search-only and write to BENCH_SEARCH_OUTPUT")
        sys.exit(1)

    st = load_state(commit, benchmark)
    st["phase"] = "searched"
    st["search_output"] = str(search_output)
    save_state(st)
    print(f"  Search results: {search_output}")
    print("  State saved. Next: --phase submit-answers")


def phase_submit_answers(benchmark: str, commit: str) -> None:
    from benchmarks.batch_api import anthropic_create_batch

    st = load_state(commit, benchmark)
    if st.get("phase") not in ("searched", "answers-submitted"):
        print(f"  State phase is '{st.get('phase')}', expected 'searched'. Run --phase search first.")
        sys.exit(1)

    search_file = st.get("search_output")
    if not search_file or not Path(search_file).exists():
        print(f"  Search output not found: {search_file}")
        sys.exit(1)

    with open(search_file) as f:
        search_data = json.load(f)

    methodology = get_methodology()
    answer_model = methodology["answer_model"]
    max_chunks = methodology["context_chunks"]

    requests = []
    for item in search_data.get("items", []):
        qid = item["id"]
        question = item["question"]
        chunks = item.get("chunks", [])
        context = "\n".join(f"- {c}" for c in chunks[:max_chunks])

        prompt = (
            "You are answering questions about a conversation based on retrieved memory snippets.\n"
            "Answer concisely in 1-3 sentences. Be direct and specific.\n"
            "If NO relevant facts exist in the memories, say "
            '"I don\'t have enough information to answer this question."\n\n'
            f"Memories:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        requests.append(
            {
                "custom_id": qid,
                "params": {
                    "model": answer_model,
                    "max_tokens": 150,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    print(f"  Submitting {len(requests)} answer requests to Anthropic batch API ({answer_model})")
    batch = anthropic_create_batch(requests)
    batch_id = batch["id"]
    print(f"  Batch created: {batch_id}")
    print(f"  Status: {batch['processing_status']}")

    st["phase"] = "answers-submitted"
    st["answer_batch_id"] = batch_id
    st["answer_model"] = answer_model
    st["total_questions"] = len(requests)
    save_state(st)
    print("  State saved. Next: --phase retrieve-answers (poll until done)")


def phase_retrieve_answers(benchmark: str, commit: str) -> None:
    from benchmarks.batch_api import anthropic_get_results, anthropic_poll_batch

    st = load_state(commit, benchmark)
    if st.get("phase") != "answers-submitted":
        print(f"  State phase is '{st.get('phase')}', expected 'answers-submitted'.")
        sys.exit(1)

    batch_id = st["answer_batch_id"]
    print(f"  Polling Anthropic batch {batch_id}...")
    anthropic_poll_batch(batch_id)

    print("  Downloading results...")
    results = anthropic_get_results(batch_id)

    answers: dict[str, str] = {}
    for r in results:
        cid = r.get("custom_id", "")
        result_data = r.get("result", {})
        if result_data.get("type") == "succeeded":
            msg = result_data.get("message", {})
            content = msg.get("content", [{}])
            text = content[0].get("text", "") if content else ""
            answers[cid] = text
        else:
            answers[cid] = ""

    answers_file = RESULTS_DIR / f"{commit}-{benchmark}-answers.json"
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent=2)

    st["phase"] = "answers-retrieved"
    st["answers_file"] = str(answers_file)
    st["answers_succeeded"] = sum(1 for v in answers.values() if v)
    save_state(st)
    print(f"  {st['answers_succeeded']}/{len(answers)} answers retrieved. Saved: {answers_file}")
    print("  Next: --phase submit-judge")


def phase_submit_judge(benchmark: str, commit: str) -> None:
    from benchmarks.batch_api import openai_create_batch, openai_upload_jsonl

    st = load_state(commit, benchmark)
    if st.get("phase") not in ("answers-retrieved", "judge-submitted"):
        print(f"  State phase is '{st.get('phase')}', expected 'answers-retrieved'.")
        sys.exit(1)

    search_file = st["search_output"]
    answers_file = st["answers_file"]

    with open(search_file) as f:
        search_data = json.load(f)
    with open(answers_file) as f:
        answers = json.load(f)

    items_by_id = {item["id"]: item for item in search_data.get("items", [])}

    judge_prompt_template = (
        "You are evaluating an AI memory system's answer.\n\n"
        "Question: {question}\n"
        "Ground Truth Answer: {gold}\n"
        "System Answer: {prediction}\n\n"
        "Is the system's answer correct? Score CORRECT only if the answer contains "
        "the specific information asked for. Score WRONG if the answer is vague, "
        "missing key facts, or incorrect. Do not give credit for answers that are "
        "technically true but don't answer the question.\n\n"
        "Reply with exactly one word: CORRECT or WRONG"
    )

    lines = []
    for qid, answer in answers.items():
        item = items_by_id.get(qid, {})
        gold = item.get("gold", "")
        question = item.get("question", "")

        prompt = judge_prompt_template.format(question=question, gold=gold, prediction=answer)
        line = json.dumps(
            {
                "custom_id": qid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 10,
                },
            }
        )
        lines.append(line)

    jsonl_content = "\n".join(lines) + "\n"
    print(f"  Uploading {len(lines)} judge requests to OpenAI...")
    file_id = openai_upload_jsonl(jsonl_content)
    print(f"  File uploaded: {file_id}")

    batch = openai_create_batch(file_id, metadata={"description": f"judge-{benchmark}-{commit[:8]}"})
    batch_id = batch["id"]
    print(f"  Batch created: {batch_id}")

    st["phase"] = "judge-submitted"
    st["judge_batch_id"] = batch_id
    st["judge_file_id"] = file_id
    save_state(st)
    print("  State saved. Next: --phase retrieve-judge (poll until done, then score)")


def phase_retrieve_judge(benchmark: str, commit: str) -> None:
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
        error = r.get("error")
        if error:
            scores[cid] = 0.0
            continue
        resp = r.get("response", {})
        if resp.get("status_code") == 200:
            body = resp.get("body", {})
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            scores[cid] = 1.0 if "CORRECT" in text.upper() else 0.0
        else:
            scores[cid] = 0.0

    total = len(scores)
    correct = sum(1 for v in scores.values() if v >= 1.0)
    accuracy = correct / total * 100 if total else 0

    methodology = get_methodology()
    final_result = {
        "commit": commit,
        "commit_short": commit[:8],
        "benchmark": benchmark,
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "methodology": methodology,
        "scores": scores,
    }

    result_file = RESULTS_DIR / f"{commit}-{benchmark}.json"
    with open(result_file, "w") as f:
        json.dump(final_result, f, indent=2)

    st["phase"] = "complete"
    st["accuracy"] = accuracy
    save_state(st)

    print(f"\n  RESULT: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Saved: {result_file}")

    save_history_row(
        {
            "commit": commit,
            "commit_short": commit[:8],
            "date": datetime.now().isoformat(),
            "benchmark": benchmark,
            "headline_score": f"{accuracy:.2f}",
            "answer_model": methodology["answer_model"],
            "judge_model": methodology["judge_model"],
            "context_chunks": str(methodology["context_chunks"]),
            "search_limit": str(methodology["search_limit"]),
            "note": "batch",
        }
    )
    print(f"  History updated: {HISTORY_FILE}")


# ─── Legacy mode (sequential) ────────────────────────────────


def run_bench_legacy(benchmark: str, extra_args: list[str], commit: str) -> dict | None:
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


# ─── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RASPUTIN benchmark runner with git integrity + batch API")
    parser.add_argument("benchmark", nargs="?", choices=list(BENCH_SCRIPTS.keys()))
    parser.add_argument("--phase", choices=PHASES, default=None, help="Run a specific batch phase")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compare-to", type=str, default=None)
    parser.add_argument("--constraints", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conversations", type=str, default=None)
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

    commit = get_commit_hash()

    if args.phase:
        is_search_phase = args.phase == "search"
        if is_search_phase and repo_is_dirty():
            print("ERROR: uncommitted changes. Commit first \u2014 benchmark must be tied to a hash.")
            sys.exit(1)

        print(f"Benchmark: {args.benchmark}")
        print(f"Phase:     {args.phase}")
        print(f"Commit:    {commit[:8]}")
        print()

        extra_args: list[str] = []
        if args.limit:
            extra_args.extend(["--limit", str(args.limit)])
        if args.conversations is not None:
            extra_args.extend(["--conversations", args.conversations])

        if args.phase == "search":
            phase_search(args.benchmark, commit, extra_args)
        elif args.phase == "submit-answers":
            phase_submit_answers(args.benchmark, commit)
        elif args.phase == "retrieve-answers":
            phase_retrieve_answers(args.benchmark, commit)
        elif args.phase == "submit-judge":
            phase_submit_judge(args.benchmark, commit)
        elif args.phase == "retrieve-judge":
            phase_retrieve_judge(args.benchmark, commit)
        sys.exit(0)

    if repo_is_dirty():
        print("ERROR: uncommitted changes. Commit first \u2014 benchmark must be tied to a hash.")
        sys.exit(1)

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

    print(f"Benchmark: {args.benchmark} (legacy sequential mode)")
    print(f"Commit:    {commit[:8]} ({commit})")
    print(f"Date:      {datetime.now().isoformat()}")
    print()

    result = run_bench_legacy(args.benchmark, extra_args, commit)
    if not result:
        print("\n  No result file produced.")
        sys.exit(1)

    methodology = get_methodology()
    result["commit"] = commit
    result["commit_short"] = commit[:8]
    result["methodology"] = methodology

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
            "answer_model": methodology["answer_model"],
            "judge_model": methodology["judge_model"],
            "context_chunks": str(methodology["context_chunks"]),
            "search_limit": str(methodology["search_limit"]),
            "note": "",
        }
    )
    print(f"  History updated: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
