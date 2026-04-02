#!/usr/bin/env python3
"""
RASPUTIN Memory — Full Hybrid Pipeline Benchmark Runner
Tests against LoCoMo benchmark through the FULL pipeline:
BM25 + vector + reranker + entity boost + keyword overlap

Usage:
    python3 benchmarks/full_pipeline_bench.py [--conversations 0,1,2] [--port 7778]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
CHECKPOINT_FILE = RESULTS_DIR / "locomo_checkpoint.json"
LOCOMO_FILE = BENCH_DIR / "locomo" / "locomo10.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:11435")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "")  # auto-detect if empty
BENCH_PORT = 7778

# LoCoMo category names
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def http_json(url, data=None, method=None, timeout=30):
    """Simple HTTP JSON helper without requests dependency."""
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def detect_vllm_model():
    """Auto-detect available vLLM model."""
    if VLLM_MODEL:
        return VLLM_MODEL
    try:
        data = http_json(f"{VLLM_URL}/v1/models")
        models = [m["id"] for m in data.get("data", [])]
        if models:
            print(f"  Detected vLLM model: {models[0]}")
            return models[0]
    except Exception as e:
        print(f"  WARNING: Could not detect vLLM model: {e}")
    return "qwen3.5-122b-a10b"


def create_qdrant_collection(name, dim=768):
    """Create a fresh Qdrant collection."""
    # Delete if exists
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    time.sleep(0.3)
    
    data = {
        "vectors": {"size": dim, "distance": "Cosine"},
        "optimizers_config": {"indexing_threshold": 0}  # index immediately
    }
    http_json(f"{QDRANT_URL}/collections/{name}", data=data, method="PUT")
    print(f"  Created collection: {name}")


def delete_qdrant_collection(name):
    """Delete a Qdrant collection."""
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def start_bench_server(collection, port=BENCH_PORT):
    """Start a temporary hybrid brain server for benchmarking."""
    env = os.environ.copy()
    env["QDRANT_COLLECTION"] = collection
    env["PORT"] = str(port)
    env["DISABLE_FALKORDB"] = "true"  # Don't need graph for benchmark
    env["PYTHONPATH"] = str(REPO / "tools")
    
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "tools" / "hybrid_brain.py"), "--port", str(port)],
        cwd=str(REPO / "tools"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to be ready
    url = f"http://localhost:{port}/health"
    for attempt in range(30):
        time.sleep(1)
        try:
            http_json(url)
            print(f"  Server ready on port {port} (collection={collection})")
            return proc
        except Exception:
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode()[-500:]
                stderr = proc.stderr.read().decode()[-500:]
                print(f"  Server died! stdout: {stdout}")
                print(f"  stderr: {stderr}")
                raise RuntimeError("Bench server died during startup")
    
    proc.kill()
    raise RuntimeError("Bench server failed to start in 30s")


def kill_server(proc):
    """Kill the benchmark server."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def get_embedding(text, prefix="search_document: "):
    """Get embedding from Ollama nomic-embed-text."""
    result = http_json(
        "http://localhost:11434/api/embed",
        data={"model": "nomic-embed-text", "input": prefix + text},
        timeout=30,
    )
    return result["embeddings"][0]


def commit_conversation_direct(conv, collection):
    """Commit conversation turns directly to Qdrant (bypasses server for speed).
    Search still goes through the full hybrid pipeline."""
    import hashlib
    
    committed = 0
    failed = 0
    points = []
    
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        date_key = f"session_{session_idx}_date_time"
        
        if session_key not in conv:
            break
        
        session_date = conv.get(date_key, f"Session {session_idx}")
        turns = conv[session_key]
        
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            if not text:
                continue
            
            commit_text = f"[{session_date}] {speaker}: {text}"
            
            try:
                vec = get_embedding(commit_text[:2000])
                point_id = int(hashlib.md5(commit_text.encode()).hexdigest()[:15], 16)
                points.append({
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "text": commit_text[:4000],
                        "source": "locomo_bench",
                        "source_weight": 1.0,
                        "date": datetime.now().isoformat(),
                        "importance": 70,
                        "retrieval_count": 0,
                    }
                })
                committed += 1
                
                # Batch upsert every 50 points
                if len(points) >= 50:
                    http_json(
                        f"{QDRANT_URL}/collections/{collection}/points",
                        data={"points": points},
                        method="PUT",
                        timeout=30,
                    )
                    points = []
                
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"    Embed/commit error: {e}")
        
        session_idx += 1
    
    # Flush remaining
    if points:
        try:
            http_json(
                f"{QDRANT_URL}/collections/{collection}/points",
                data={"points": points},
                method="PUT",
                timeout=30,
            )
        except Exception as e:
            print(f"    Final batch error: {e}")
            failed += len(points)
            committed -= len(points)
    
    print(f"  Committed {committed} turns ({failed} failed) across {session_idx - 1} sessions")
    
    # Wait for indexing
    time.sleep(2)
    return committed


def search_query(query, port=BENCH_PORT, limit=10):
    """Search the benchmark server."""
    url = f"http://localhost:{port}/search"
    params = urllib.parse.urlencode({"q": query, "limit": limit})
    try:
        result = http_json(f"{url}?{params}", timeout=30)
        return result.get("results", [])
    except Exception as e:
        print(f"    Search error: {e}")
        return []


def generate_answer(question, context_chunks, model):
    """Generate answer using vLLM."""
    context = "\n\n".join(
        f"[Memory {i+1}]: {c.get('text', '')}" 
        for i, c in enumerate(context_chunks[:10])
    )
    
    prompt = f"""Answer the question using ONLY the memories below. Give the shortest possible answer (a few words or a short phrase). Do NOT explain or add context.

Memories:
{context}

Question: {question}
Answer (short):"""

    try:
        result = http_json(
            f"{VLLM_URL}/v1/chat/completions",
            data={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        msg = result["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        return content.strip()
    except Exception as e:
        print(f"    LLM error: {e}")
        return ""


def compute_f1(prediction, ground_truth):
    """Token-level F1 score (LoCoMo paper method)."""
    pred_tokens = set(re.findall(r'\w+', prediction.lower()))
    truth_tokens = set(re.findall(r'\w+', str(ground_truth).lower()))
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    return 2 * precision * recall / (precision + recall)


def run_benchmark(conversations, conv_indices=None, port=BENCH_PORT):
    """Run the full pipeline benchmark on LoCoMo conversations."""
    model = detect_vllm_model()
    
    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
    
    all_results = checkpoint.get("results", {})
    
    if conv_indices is None:
        conv_indices = list(range(len(conversations)))
    
    for idx in conv_indices:
        conv_data = conversations[idx]
        conv_id = conv_data.get("sample_id", f"conv-{idx}")
        
        if conv_id in all_results:
            print(f"\n[{idx+1}/{len(conv_indices)}] Skipping {conv_id} (already done)")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(conv_indices)}] Processing {conv_id}")
        print(f"{'='*60}")
        
        collection = f"locomo_bench_{conv_id.replace('-', '_')}"
        proc = None
        
        try:
            # Step 1: Create collection
            create_qdrant_collection(collection)
            
            # Step 2: Start server
            proc = start_bench_server(collection, port)
            
            # Step 3: Commit conversation directly to Qdrant (fast)
            n_committed = commit_conversation_direct(conv_data["conversation"], collection)
            
            # Step 4: Run QA
            qa_results = []
            qa_list = conv_data.get("qa", [])
            print(f"  Running {len(qa_list)} questions...")
            
            for qi, qa in enumerate(qa_list):
                question = qa["question"]
                ground_truth = str(qa.get("answer", qa.get("adversarial_answer", "")))
                category = qa.get("category", 0)
                
                if not ground_truth:
                    continue
                
                # Search
                chunks = search_query(question, port)
                
                # Generate
                prediction = generate_answer(question, chunks, model)
                
                # Score
                f1 = compute_f1(prediction, ground_truth)
                
                qa_results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "f1": f1,
                    "category": category,
                    "n_chunks": len(chunks),
                })
                
                # Rate limit: max ~60/min for search
                time.sleep(0.6)
                
                if (qi + 1) % 20 == 0:
                    avg_so_far = sum(r["f1"] for r in qa_results) / len(qa_results)
                    print(f"    Progress: {qi+1}/{len(qa_list)} questions, running F1={avg_so_far:.4f}")
            
            # Compute stats
            conv_f1 = sum(r["f1"] for r in qa_results) / len(qa_results) if qa_results else 0
            
            cat_scores = defaultdict(list)
            for r in qa_results:
                cat_scores[r["category"]].append(r["f1"])
            
            cat_f1 = {
                cat: sum(scores) / len(scores) 
                for cat, scores in sorted(cat_scores.items())
            }
            
            all_results[conv_id] = {
                "conv_id": conv_id,
                "n_questions": len(qa_results),
                "n_committed": n_committed,
                "mean_f1": conv_f1,
                "category_f1": cat_f1,
                "details": qa_results,
            }
            
            print(f"  ✅ {conv_id}: F1={conv_f1:.4f} ({len(qa_results)} questions)")
            for cat, score in sorted(cat_f1.items()):
                cat_name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
                print(f"     {cat_name}: {score:.4f} ({len(cat_scores[cat])} Qs)")
            
            # Save checkpoint
            checkpoint["results"] = all_results
            checkpoint["last_updated"] = datetime.now().isoformat()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint, f, indent=2)
        
        except Exception as e:
            print(f"  ❌ Error on {conv_id}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            kill_server(proc)
            delete_qdrant_collection(collection)
            time.sleep(1)
    
    return all_results


def generate_report(all_results):
    """Generate markdown report."""
    if not all_results:
        return "No results."
    
    # Overall stats
    all_f1s = []
    cat_all = defaultdict(list)
    
    for conv_id, data in all_results.items():
        for r in data.get("details", []):
            all_f1s.append(r["f1"])
            cat_all[r["category"]].append(r["f1"])
    
    overall_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0
    
    # Leaderboard
    leaderboard = [
        ("Backboard", 90.00),
        ("Memvid", 85.70),
        ("MemMachine", 84.87),
        ("Memobase", 75.78),
        ("Zep", 75.14),
        ("mem0", 66.88),
        ("RASPUTIN raw vector", 41.44),
    ]
    
    # Insert RASPUTIN full pipeline
    rasputin_score = round(overall_f1 * 100, 2)
    leaderboard.append(("RASPUTIN full pipeline", rasputin_score))
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    
    lines = []
    lines.append("# RASPUTIN Memory — LoCoMo Full Pipeline Benchmark Results")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Pipeline:** BM25 + vector + reranker + entity boost + keyword overlap")
    lines.append(f"**Conversations:** {len(all_results)}")
    lines.append(f"**Total questions:** {len(all_f1s)}")
    lines.append(f"\n## Overall F1: {rasputin_score:.2f}")
    
    lines.append("\n### Improvement over raw vector")
    lines.append("- Raw vector: 41.44")
    lines.append(f"- Full pipeline: {rasputin_score:.2f}")
    improvement = rasputin_score - 41.44
    lines.append(f"- **Improvement: +{improvement:.2f} ({improvement/41.44*100:.1f}%)**")
    
    lines.append("\n## Leaderboard")
    for i, (name, score) in enumerate(leaderboard, 1):
        marker = " ← **YOU ARE HERE**" if "full pipeline" in name else ""
        lines.append(f"{i}. **{name}**: {score:.2f}{marker}")
    
    lines.append("\n## Per-Category Breakdown")
    for cat in sorted(cat_all.keys()):
        scores = cat_all[cat]
        cat_name = CATEGORY_NAMES.get(cat, f"category-{cat}")
        avg = sum(scores) / len(scores) if scores else 0
        lines.append(f"- **{cat_name}** (cat {cat}): {avg*100:.2f} ({len(scores)} questions)")
    
    lines.append("\n## Per-Conversation Results")
    for conv_id, data in sorted(all_results.items()):
        lines.append(f"- **{conv_id}**: F1={data['mean_f1']*100:.2f} ({data['n_questions']} Qs, {data['n_committed']} committed)")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RASPUTIN Full Pipeline LoCoMo Benchmark")
    parser.add_argument("--conversations", type=str, default=None,
                        help="Comma-separated indices (e.g. 0,1,2). Default: all")
    parser.add_argument("--port", type=int, default=BENCH_PORT)
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    args = parser.parse_args()
    
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared.")
    
    # Load dataset
    print(f"Loading LoCoMo dataset from {LOCOMO_FILE}...")
    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")
    
    conv_indices = None
    if args.conversations:
        conv_indices = [int(x) for x in args.conversations.split(",")]
    
    # Run
    results = run_benchmark(conversations, conv_indices, args.port)
    
    # Report
    report = generate_report(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "locomo-fullpipeline-results.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
