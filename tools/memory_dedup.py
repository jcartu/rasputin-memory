#!/usr/bin/env python3
"""
Memory Deduplication Engine — Phase 3A of RASPUTIN Intelligence Layer.

Scans Qdrant second_brain collection in batches, finds near-duplicate clusters
(cosine similarity > 0.92), keeps the best one per cluster, deletes the rest.

SAFE by default: dry-run mode. Use --execute to actually delete.
RESUMABLE: saves progress to checkpoint file.

Usage:
    python3 memory_dedup.py                    # Dry run, scan all
    python3 memory_dedup.py --limit 5000       # Dry run, first 5K vectors
    python3 memory_dedup.py --execute          # Actually delete duplicates
    python3 memory_dedup.py --threshold 0.95   # Stricter similarity threshold
    python3 memory_dedup.py --resume           # Resume from checkpoint
"""

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from pipeline.dateparse import parse_date
from pipeline.scoring_constants import SOURCE_IMPORTANCE

safe_import = importlib.import_module("pipeline._imports").safe_import

_locking = safe_import("pipeline.locking", "tools.pipeline.locking")
acquire_pipeline_lock = _locking.acquire_lock

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".dedup_checkpoint.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), ".dedup_log.jsonl")
qdrant = QdrantClient(url=QDRANT_URL)


def load_checkpoint():
    """Load checkpoint for resumable scanning."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_offset": None, "scanned": 0, "clusters_found": 0, "dupes_marked": 0, "processed_ids": []}


def save_checkpoint(state):
    """Save checkpoint for resumable scanning."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")


def log_action(action_type, data):
    """Append action to log file."""
    try:
        entry = {"ts": datetime.now().isoformat(), "action": action_type, **data}
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def score_memory(payload):
    """Score a memory for quality — higher is better. Used to pick the 'best' in a cluster."""
    score = 0
    text = payload.get("text", "")

    # Length: longer text = more specific (up to a point)
    score += min(len(text), 2000) / 100  # max 20 points

    # Source quality
    source = str(payload.get("source", "")).strip().lower()
    source_priority = int(SOURCE_IMPORTANCE.get(source, 0.5) * 10)
    if source_priority == int(0.5 * 10) and "social_intel" in source:
        source_priority = int(SOURCE_IMPORTANCE["social_intel"] * 10)
    score += source_priority

    # Importance
    imp = payload.get("importance", 50)
    try:
        imp = int(imp) if imp is not None else 50
    except (ValueError, TypeError):
        imp = 50
    score += imp / 10  # max 10 points

    # Recency: newer is better
    date_str = payload.get("date", "")
    if date_str:
        dt = parse_date(date_str)
        if dt is not None:
            days_old = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            score += max(0, 10 - days_old / 30)  # max 10 points, decays over months

    # Retrieval count: frequently accessed = valuable
    ret_count = payload.get("retrieval_count", 0) or 0
    score += min(ret_count, 10)  # max 10 points

    # Has graph connections
    if payload.get("connected_to"):
        score += 5

    return round(score, 2)


def find_duplicates_for_point(point_id, vector, threshold=0.92, limit=10):
    """Find near-duplicate neighbors for a given point."""
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=limit + 1,  # +1 because the point itself will be in results
            with_payload=True,
        )
        dupes = []
        for p in results.points:
            if p.id == point_id:
                continue
            if p.score >= threshold:
                dupes.append(
                    {
                        "id": p.id,
                        "score": round(p.score, 4),
                        "payload": dict(p.payload) if p.payload else {},
                    }
                )
        return dupes
    except Exception as e:
        print(f"[ERROR] Search failed for point {point_id}: {e}")
        return []


def mark_pending_delete(point_ids):
    if not point_ids:
        return
    try:
        qdrant.set_payload(
            collection_name=COLLECTION,
            points=point_ids,
            payload={
                "pending_delete": True,
                "pending_delete_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        print(f"  [WARN] Failed to mark pending_delete for batch: {e}")


def run_dedup(threshold=0.92, limit=None, execute=False, resume=False, batch_size=100):
    """Main deduplication loop."""
    print(f"{'=' * 60}")
    print("RASPUTIN Memory Deduplication Engine")
    print(f"{'=' * 60}")
    print(f"Mode: {'🔴 EXECUTE (will delete!)' if execute else '🟢 DRY RUN (safe)'}")
    print(f"Threshold: {threshold}")
    print(f"Limit: {limit or 'all'}")
    print(f"Batch size: {batch_size}")
    print()

    # Check Qdrant
    try:
        info = qdrant.get_collection(COLLECTION)
        total_points = info.points_count
        print(f"Collection: {COLLECTION} ({total_points:,} points)")
    except Exception as e:
        print(f"[FATAL] Cannot connect to Qdrant: {e}")
        sys.exit(1)

    # Load or reset checkpoint
    if resume:
        state = load_checkpoint()
        print(f"Resuming from checkpoint: scanned={state['scanned']}, clusters={state['clusters_found']}")
    else:
        state = {"last_offset": None, "scanned": 0, "clusters_found": 0, "dupes_marked": 0, "processed_ids": []}

    # Track which IDs we've already processed (to avoid double-counting)
    processed_set = set(state.get("processed_ids", []))
    # Track which IDs are marked for deletion (to avoid deleting the same thing twice)
    to_delete = set()
    # Cluster details for reporting
    clusters = []

    scan_limit = limit or total_points
    offset = state["last_offset"]

    start_time = time.time()
    last_report = time.time()

    while state["scanned"] < scan_limit:
        # Scroll batch
        try:
            scroll_result = qdrant.scroll(
                collection_name=COLLECTION,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            points, next_offset = scroll_result
        except Exception as e:
            print(f"[ERROR] Scroll failed at offset {offset}: {e}")
            save_checkpoint(state)
            break

        if not points:
            print(f"\n[INFO] Reached end of collection at {state['scanned']} points.")
            break

        for point in points:
            if state["scanned"] >= scan_limit:
                break

            pid = point.id
            state["scanned"] += 1

            # Skip if already processed or marked for deletion
            if pid in processed_set or pid in to_delete:
                continue

            processed_set.add(pid)

            # Find duplicates
            dupes = find_duplicates_for_point(pid, point.vector, threshold=threshold)

            if not dupes:
                continue

            # Build cluster: this point + its dupes
            cluster_members = [{"id": pid, "payload": dict(point.payload) if point.payload else {}}]
            for d in dupes:
                if d["id"] not in to_delete:
                    cluster_members.append(d)

            if len(cluster_members) < 2:
                continue

            # Score each member, keep the best
            for m in cluster_members:
                m["quality_score"] = score_memory(m["payload"])

            cluster_members.sort(key=lambda x: x["quality_score"], reverse=True)
            keeper = cluster_members[0]
            removable = cluster_members[1:]

            state["clusters_found"] += 1

            for r in removable:
                if r["id"] not in to_delete:
                    to_delete.add(r["id"])
                    state["dupes_marked"] += 1

            clusters.append(
                {
                    "keeper_id": keeper["id"],
                    "keeper_score": keeper["quality_score"],
                    "keeper_text": keeper["payload"].get("text", "")[:80],
                    "removed": len(removable),
                    "removed_ids": [r["id"] for r in removable],
                }
            )

            log_action(
                "cluster",
                {
                    "keeper": keeper["id"],
                    "removed": [r["id"] for r in removable],
                    "similarity": dupes[0]["score"] if dupes else 0,
                },
            )

        # Progress reporting every 30s
        if time.time() - last_report > 30:
            elapsed = time.time() - start_time
            rate = state["scanned"] / elapsed if elapsed > 0 else 0
            print(
                f"  Scanned: {state['scanned']:,}/{scan_limit:,} | "
                f"Clusters: {state['clusters_found']} | "
                f"Dupes: {state['dupes_marked']} | "
                f"Rate: {rate:.0f}/s | "
                f"Elapsed: {elapsed:.0f}s"
            )
            last_report = time.time()

        # Save checkpoint after each batch
        offset = next_offset
        state["last_offset"] = offset
        # Don't save full processed_ids to checkpoint (too large), just the offset
        save_checkpoint(state)

        if offset is None:
            print("\n[INFO] Reached end of collection.")
            break

    # Final report
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("DEDUP RESULTS")
    print(f"{'=' * 60}")
    print(f"Scanned: {state['scanned']:,} vectors")
    print(f"Duplicate clusters found: {state['clusters_found']}")
    print(f"Vectors marked for removal: {state['dupes_marked']}")
    print(f"Time: {elapsed:.1f}s")

    if clusters:
        print("\nTop 10 clusters (by size):")
        top = sorted(clusters, key=lambda x: x["removed"], reverse=True)[:10]
        for i, c in enumerate(top):
            print(
                f"  {i + 1}. Keeper: {c['keeper_id']} (score={c['keeper_score']}) | "
                f"Removed: {c['removed']} dupes | "
                f"Text: {c['keeper_text']}..."
            )

    # Execute deletions
    if execute and to_delete:
        print(f"\n🔴 EXECUTING: Deleting {len(to_delete)} duplicate vectors...")
        delete_list = list(to_delete)
        batch_size_del = 100
        deleted = 0
        for i in range(0, len(delete_list), batch_size_del):
            batch = delete_list[i : i + batch_size_del]
            try:
                mark_pending_delete(batch)
                qdrant.delete(
                    collection_name=COLLECTION,
                    points_selector=PointIdsList(points=batch),
                )
                deleted += len(batch)
                print(f"  Deleted batch {i // batch_size_del + 1}: {len(batch)} vectors (total: {deleted})")
                log_action("delete", {"count": len(batch), "ids": batch[:5]})
            except Exception as e:
                print(f"  [ERROR] Delete failed at batch {i // batch_size_del + 1}: {e}")
                break
        print(f"\n✅ Deleted {deleted}/{len(to_delete)} duplicate vectors.")
    elif to_delete:
        print(f"\n🟢 DRY RUN: Would delete {len(to_delete)} vectors. Use --execute to proceed.")
    else:
        print("\n✅ No duplicates found.")

    # Clean up checkpoint on completion
    if not limit or state["scanned"] >= scan_limit:
        try:
            os.remove(CHECKPOINT_FILE)
        except Exception:
            pass

    return state


if __name__ == "__main__":
    try:
        _lock_fd = acquire_pipeline_lock("memory_dedup")
    except OSError:
        print("[INFO] Another memory_dedup instance is running. Exiting.")
        sys.exit(0)
    parser = argparse.ArgumentParser(description="RASPUTIN Memory Deduplication Engine")
    parser.add_argument("--threshold", type=float, default=0.92, help="Cosine similarity threshold (default: 0.92)")
    parser.add_argument("--limit", type=int, default=None, help="Max vectors to scan (default: all)")
    parser.add_argument("--execute", action="store_true", help="Actually delete duplicates (default: dry run)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=100, help="Scroll batch size (default: 100)")
    args = parser.parse_args()

    run_dedup(
        threshold=args.threshold,
        limit=args.limit,
        execute=args.execute,
        resume=args.resume,
        batch_size=args.batch_size,
    )
