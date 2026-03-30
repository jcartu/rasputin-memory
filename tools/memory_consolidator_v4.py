#!/usr/bin/env python3
"""
MEMORY CONSOLIDATOR v4 — MAX PARALLEL
Parallel workers using configurable LLM endpoints.
Large context windows per session (30K chars), fresh dedup (no old hashes).
Resume-safe.
"""

import fcntl
import json, os, sys, time, hashlib, glob, datetime, requests, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# === CONFIG ===
SESSIONS_DIR = os.environ.get("SESSIONS_DIR", os.path.expanduser("~/.openclaw/agents/main/sessions"))
OUT_DIR = os.environ.get("CONSOLIDATION_DIR", os.path.expanduser("./memory/consolidation"))
FACTS_FILE = os.environ.get("FACTS_FILE", os.path.expanduser("./memory/facts.jsonl"))
BRAIN_URL = os.environ.get("MEMORY_API_URL", "http://localhost:7777")
PROGRESS_FILE = os.path.join(OUT_DIR, "progress-v4.json")
OUTPUT_FILE = os.path.join(OUT_DIR, "extracted-v4.jsonl")
SUMMARY_FILE = os.path.join(OUT_DIR, "CONSOLIDATION-REPORT-v4.md")
LOG_FILE = os.path.join(OUT_DIR, "consolidator-v4.log")
LOCK_FILE = "/tmp/rasputin_memory_consolidator_v4.lock"

# 8 workers total (4x 122B + 4x 35B) — leave headroom for main session
LLM_ENDPOINTS = []
# 2x 122B direct GPU0 (port 11435) + 2x 122B direct GPU2 (port 11437)
# Direct connections bypass llama-swap 502 errors
for i in range(2):
    LLM_ENDPOINTS.append(
        {
            "url": "http://localhost:11435/v1/chat/completions",
            "model": "qwen3.5:122b",
            "name": f"122B-gpu0-s{i}",
            "max_tokens": 8192,
        }
    )
for i in range(2):
    LLM_ENDPOINTS.append(
        {
            "url": "http://localhost:11437/v1/chat/completions",
            "model": "qwen3.5:122b",
            "name": f"122B-gpu2-s{i}",
            "max_tokens": 8192,
        }
    )
# 4x 35B on local 5090
for i in range(4):
    LLM_ENDPOINTS.append(
        {
            "url": "http://localhost:5800/v1/chat/completions",
            "model": "qwen3.5:35b",
            "name": f"5090-35B-s{i}",
            "max_tokens": 4096,
        }
    )

MAX_WORKERS = 8
MAX_CHARS_PER_SESSION = 30000  # Much bigger than v3's 6K
SESSIONS_PER_BATCH = 1  # 1 session per call for bigger context

os.makedirs(OUT_DIR, exist_ok=True)

# Thread-safe
log_lock = threading.Lock()
progress_lock = threading.Lock()
output_lock = threading.Lock()
hash_lock = threading.Lock()
stats_lock = threading.Lock()

# Shared state
seen_hashes = set()
stats = {"sessions": 0, "facts": 0, "committed": 0, "skipped": 0, "errors": 0}
processed_set = set()


def acquire_lock():
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except OSError:
        print("[INFO] Another memory_consolidator_v4 instance is running. Exiting.")
        sys.exit(0)


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
        # Also stdout for debugging
        print(line, flush=True)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "processed": [],
        "stats": {"sessions": 0, "facts": 0, "committed": 0, "skipped": 0, "errors": 0},
        "start_time": time.time(),
    }


def save_progress():
    with progress_lock:
        prog = {"processed": list(processed_set), "stats": dict(stats), "start_time": start_time}
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(prog, f)
        os.replace(tmp, PROGRESS_FILE)


def extract_text(content_raw):
    if isinstance(content_raw, str):
        return content_raw.strip()
    if isinstance(content_raw, list):
        parts = []
        for part in content_raw:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text", "")
                if t:
                    parts.append(t)
        return " ".join(parts).strip()
    return ""


def read_session(path, max_chars=MAX_CHARS_PER_SESSION):
    """Read session with much larger context window."""
    messages = []
    total_chars = 0
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except:
                    continue
                if entry.get("type") != "message":
                    continue
                msg = entry.get("message", {})
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                text = extract_text(msg.get("content", ""))
                # Skip very short, tool results, heartbeat noise
                if len(text) < 30:
                    continue
                if "HEARTBEAT_OK" in text or "NO_REPLY" in text:
                    continue
                if text.startswith("✅ New session started"):
                    continue
                # Skip tool call JSON blobs
                if text.startswith("{") and '"type"' in text[:50]:
                    continue

                # Truncate individual messages but keep more
                if len(text) > 2000:
                    text = text[:2000] + "…"

                entry_text = f"[{role.upper()}]: {text}"
                if total_chars + len(entry_text) > max_chars:
                    break
                messages.append(entry_text)
                total_chars += len(entry_text)
    except Exception as e:
        return None

    if len(messages) < 3:
        return None
    return "\n".join(messages)


EXTRACTION_PROMPT = """You are extracting SPECIFIC, FACTUAL knowledge from a conversation between a user and their AI assistant.

Output ONLY JSON lines. One fact per line. No markdown, no commentary, no wrapping.

Categories: DECISION, TASK_DONE, PREFERENCE, PERSON, BUSINESS, TECHNICAL, PERSONAL, EVENT, INSIGHT

Format per line:
{"category":"<CAT>","fact":"<specific fact with names/numbers/dates>","date":"<YYYY-MM-DD or null>"}

Rules:
- SPECIFIC: include exact numbers, names, ports, paths, versions, amounts
- NO generic observations ("the user uses AI" = useless)
- NO meta-commentary about the conversation itself
- NO duplicates within your output
- Skip if session is just heartbeats, tool noise, or greetings
- If truly nothing valuable: output single line {"category":"SKIP","fact":"no extractable knowledge","date":null}

CONVERSATION:
"""


def call_llm(endpoint, session_text, worker_id):
    prompt = EXTRACTION_PROMPT + session_text
    try:
        payload = {
            "model": endpoint["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.15,
            "max_tokens": endpoint["max_tokens"],
            "stream": False,
        }
        # Add thinking disable for 122B
        if "122B" in endpoint["name"]:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        resp = requests.post(endpoint["url"], json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip thinking
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        facts = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                f = json.loads(line)
                if f.get("category") not in (None, "SKIP") and f.get("fact") and len(f["fact"]) > 15:
                    facts.append(f)
            except:
                continue
        return facts
    except requests.exceptions.Timeout:
        log(f"  [W{worker_id}:{endpoint['name']}] ⏰ TIMEOUT (300s)")
        return []
    except Exception as e:
        err = str(e)[:120]
        log(f"  [W{worker_id}:{endpoint['name']}] ⚠️ {err}")
        return []


def fact_hash(text):
    return hashlib.md5(text.lower().strip()[:200].encode()).hexdigest()


def commit_to_brain(fact_text, category):
    """Commit consolidated fact through hybrid_brain /commit API."""
    try:
        resp = requests.post(
            "http://localhost:7777/commit",
            json={
                "text": f"[{category}] {fact_text}",
                "source": "consolidator-v4",
                "importance": 60,
                "metadata": {"category": category, "type": "fact"},
            },
            timeout=30,
        )
        data = resp.json()
        return bool(data.get("ok"))
    except Exception as e:
        log(f"  [Consolidator-v4] Commit error: {e}")
        return False


def process_session(fpath, endpoint, worker_id):
    fname = os.path.basename(fpath)
    fsize = os.path.getsize(fpath)

    # 35B has 32K context — cap at 8K chars; 122B gets full 30K
    char_limit = 8000 if "35B" in endpoint["name"] else MAX_CHARS_PER_SESSION
    text = read_session(fpath, max_chars=char_limit)
    if not text:
        with progress_lock:
            processed_set.add(fpath)
        with stats_lock:
            stats["skipped"] += 1
        return 0

    log(f"  [W{worker_id}:{endpoint['name']}] {fname[:12]}… ({fsize // 1024}KB, {len(text) // 1000}k chars)")

    facts = call_llm(endpoint, text, worker_id)

    new_count = 0
    for fact in facts:
        ftext = fact["fact"]
        h = fact_hash(ftext)

        with hash_lock:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

        record = {
            "ts": datetime.datetime.now().isoformat(),
            "category": fact.get("category", "UNKNOWN"),
            "fact": ftext,
            "date": fact.get("date"),
            "source": "consolidator-v4",
            "hash": h,
            "session": fname,
        }
        with output_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Commit to Qdrant with rate limiting (max 2/sec, pause every 100)
        try:
            if commit_to_brain(ftext, record["category"]):
                with stats_lock:
                    stats["committed"] += 1
                    if stats["committed"] % 100 == 0:
                        log(f"  📊 Committed {stats['committed']} facts so far...")
                        time.sleep(5)  # batch pause every 100 commits
            time.sleep(0.5)  # rate limit: max 2 commits/sec
        except:
            pass

        new_count += 1
        with stats_lock:
            stats["facts"] += 1

    with progress_lock:
        processed_set.add(fpath)
    with stats_lock:
        stats["sessions"] = len(processed_set)

    if new_count > 0:
        log(f"  [W{worker_id}:{endpoint['name']}] ✅ +{new_count} facts (total: {stats['facts']})")

    return new_count


def main():
    _lock_fd = acquire_lock()
    global start_time, seen_hashes, processed_set, stats

    # Clear old log
    open(LOG_FILE, "w").close()

    log("=" * 60)
    log("🧠 MEMORY CONSOLIDATOR v4 — MAX PARALLEL")
    log(f"   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M MSK')}")
    log(f"   Workers: {MAX_WORKERS} (4x 122B + 4x 5090-35B)")
    log(f"   Max chars/session: {MAX_CHARS_PER_SESSION // 1000}K")
    log("=" * 60)

    # Load progress (only from v4)
    prog = load_progress()
    processed_set = set(prog["processed"])
    stats = prog["stats"]
    start_time = prog.get("start_time", time.time())

    # Only dedup against THIS run's output (not old facts.jsonl)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen_hashes.add(d.get("hash", fact_hash(d.get("fact", ""))))
                except:
                    pass
    log(f"  Existing v4 hashes: {len(seen_hashes)}")

    # Sort sessions by size DESC — big sessions first (most knowledge)
    all_files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")), key=os.path.getsize, reverse=True)
    remaining = [f for f in all_files if f not in processed_set]

    log(f"  Total sessions: {len(all_files)}")
    log(f"  Already done: {len(processed_set)}")
    log(f"  Remaining: {len(remaining)}")

    # Size distribution
    big = sum(1 for f in remaining if os.path.getsize(f) > 50000)
    med = sum(1 for f in remaining if 5000 <= os.path.getsize(f) <= 50000)
    small = sum(1 for f in remaining if os.path.getsize(f) < 5000)
    log(f"  Big (>50KB): {big} | Med (5-50KB): {med} | Small (<5KB): {small}")

    batch_idx = [0]
    idx_lock = threading.Lock()
    save_counter = [0]

    def get_next():
        with idx_lock:
            i = batch_idx[0]
            batch_idx[0] += 1
            return i

    def worker(wid):
        endpoint = LLM_ENDPOINTS[wid]
        while True:
            idx = get_next()
            if idx >= len(remaining):
                break
            fpath = remaining[idx]
            try:
                process_session(fpath, endpoint, wid)
            except Exception as e:
                log(f"  [W{wid}] ❌ {str(e)[:100]}")
                with stats_lock:
                    stats["errors"] += 1

            # Save every 10 sessions
            with idx_lock:
                save_counter[0] += 1
                if save_counter[0] % 10 == 0:
                    save_progress()
                    pct = int(len(processed_set) / len(all_files) * 100)
                    elapsed_min = (time.time() - start_time) / 60
                    rate = len(processed_set) / max(elapsed_min, 0.1)
                    eta_min = (len(remaining) - batch_idx[0]) / max(rate, 0.1)
                    log(
                        f"  📊 {len(processed_set)}/{len(all_files)} ({pct}%) | Facts: {stats['facts']} | Brain: {stats['committed']} | ETA: {eta_min:.0f}min"
                    )

    log(f"\n🚀 Launching {MAX_WORKERS} workers...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, i) for i in range(MAX_WORKERS)]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                log(f"  ❌ Worker died: {e}")

    save_progress()
    elapsed = time.time() - start_time

    # Category breakdown
    cats = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    c = r.get("category", "?")
                    cats[c] = cats.get(c, 0) + 1
                except:
                    pass

    report = f"""# 🧠 Memory Consolidation Report — v4 MAX PARALLEL
*{datetime.datetime.now().strftime("%Y-%m-%d %H:%M MSK")}*

## Summary
| Metric | Value |
|--------|-------|
| Sessions processed | {len(processed_set)} / {len(all_files)} |
| New facts extracted | {stats["facts"]} |
| Committed to Qdrant | {stats["committed"]} |
| Skipped (trivial) | {stats["skipped"]} |
| Errors | {stats["errors"]} |
| Workers | {MAX_WORKERS} (6x 122B + 6x 5090-35B) |
| Runtime | {elapsed / 3600:.1f}h ({elapsed / 60:.0f}min) |
| Throughput | {len(processed_set) / (elapsed / 60):.1f} sessions/min |

## Facts by Category
"""
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        report += f"- **{cat}**: {count}\n"

    report += "\n## Sample Facts (last 100)\n"
    if os.path.exists(OUTPUT_FILE):
        lines = open(OUTPUT_FILE).readlines()
        for line in lines[-100:]:
            try:
                r = json.loads(line)
                report += f"- `[{r['category']}]` {r['fact']}\n"
            except:
                pass

    with open(SUMMARY_FILE, "w") as f:
        f.write(report)

    log("")
    log("=" * 60)
    log(f"✅ DONE — {stats['facts']} facts, {stats['committed']} committed, {elapsed / 60:.0f}min")
    log("=" * 60)


if __name__ == "__main__":
    main()
