#!/usr/bin/env python3
"""
Memory Decay Engine — Phase 3B of RASPUTIN Intelligence Layer.

Importance-based memory lifecycle management:
- Memories not accessed in 90 days with low importance → archive
- Memories not accessed in 180 days → soft delete (exclude from search)
- NEVER permanently deletes — always archives first.

Run weekly via cron.

Usage:
    python3 memory_decay.py                # Dry run
    python3 memory_decay.py --execute      # Actually archive/soft-delete
    python3 memory_decay.py --stats        # Show memory age distribution
"""

import argparse
import fcntl
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    PointIdsList,
    Filter,
    FieldCondition,
    MatchValue,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")
ARCHIVE_COLLECTION = "memories_archive"
LOCK_FILE = "/tmp/rasputin_memory_decay.lock"

# Thresholds
ARCHIVE_DAYS = 90  # Not accessed in 90 days + low importance → archive
SOFT_DELETE_DAYS = 180  # Not accessed in 180 days → soft delete
LOW_IMPORTANCE_THRESHOLD = 40  # Below this = "low importance"

qdrant = QdrantClient(url=QDRANT_URL)


def acquire_lock():
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except OSError:
        print("[INFO] Another memory_decay instance is running. Exiting.")
        sys.exit(0)


def ensure_archive_collection():
    """Create archive collection if it doesn't exist."""
    try:
        qdrant.get_collection(ARCHIVE_COLLECTION)
    except Exception:
        try:
            # Get vector config from source collection
            source_info = qdrant.get_collection(COLLECTION)
            vectors_config = source_info.config.params.vectors
            qdrant.create_collection(
                collection_name=ARCHIVE_COLLECTION,
                vectors_config=vectors_config,
            )
            print(f"[INFO] Created archive collection: {ARCHIVE_COLLECTION}")
        except Exception as e:
            print(f"[ERROR] Failed to create archive collection: {e}")
            sys.exit(1)


def compute_importance_score(payload):
    """Compute importance score for a memory (0-100 scale)."""
    score = 0

    # Base importance from payload
    imp = payload.get("importance", 50)
    try:
        imp = int(imp) if imp is not None else 50
    except (ValueError, TypeError):
        imp = 50
    score += imp * 0.4  # 40% weight

    # Source quality
    source_weights = {
        "manual_commit": 25,
        "fact_extraction": 20,
        "fact_extractor": 20,
        "conversation": 12,
        "chatgpt": 10,
        "perplexity": 8,
        "email": 8,
        "telegram": 8,
        "web_page": 5,
        "social_intel": 3,
        "benchmark_test": 0,
    }
    source = payload.get("source", "")
    source_score = source_weights.get(source, 8)
    # Handle partial matches
    if source_score == 8 and "social_intel" in source:
        source_score = 3
    score += source_score

    # Content quality heuristics
    text = payload.get("text", "")
    # Business/personal facts with numbers, names, dates
    import re

    has_numbers = bool(re.search(r"\$[\d,]+|\d+%|\d+K|\€[\d,]+", text))
    has_names = bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text))
    has_dates = bool(re.search(r"\d{4}-\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", text))

    if has_numbers:
        score += 8
    if has_names:
        score += 5
    if has_dates:
        score += 3

    # Text length (more specific = better)
    if len(text) > 200:
        score += 5
    elif len(text) > 50:
        score += 2

    # Retrieval count (frequently accessed = important)
    ret_count = payload.get("retrieval_count", 0) or 0
    score += min(ret_count * 2, 15)

    # Graph connections
    if payload.get("connected_to"):
        connections = payload["connected_to"]
        if isinstance(connections, list):
            score += min(len(connections) * 3, 10)
        else:
            score += 5

    return min(round(score, 1), 100)


def get_last_accessed(payload):
    """Get last accessed datetime — use last_accessed field or fall back to date."""
    la = payload.get("last_accessed")
    if la:
        try:
            return datetime.fromisoformat(la[:26])
        except Exception:
            pass

    # Fall back to creation date
    date_str = payload.get("date", "")
    if date_str:
        try:
            return datetime.fromisoformat(date_str[:26])
        except Exception:
            pass

    return None


def scan_memories(limit=None):
    """Scan all memories and categorize by decay status."""
    now = datetime.now()

    stats = {
        "total": 0,
        "active": 0,
        "archive_candidates": 0,
        "soft_delete_candidates": 0,
        "protected_high_importance": 0,
        "no_date": 0,
    }

    archive_candidates = []
    soft_delete_candidates = []
    age_distribution = defaultdict(int)

    offset = None
    batch_size = 100

    while True:
        try:
            points, next_offset = qdrant.scroll(
                collection_name=COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"[ERROR] Scroll failed: {e}")
            break

        if not points:
            break

        for point in points:
            stats["total"] += 1
            payload = dict(point.payload) if point.payload else {}

            last_accessed = get_last_accessed(payload)
            if not last_accessed:
                stats["no_date"] += 1
                continue

            days_since_access = (now - last_accessed).total_seconds() / 86400
            importance = compute_importance_score(payload)

            # Age distribution
            if days_since_access < 7:
                age_distribution["<1 week"] += 1
            elif days_since_access < 30:
                age_distribution["1-4 weeks"] += 1
            elif days_since_access < 90:
                age_distribution["1-3 months"] += 1
            elif days_since_access < 180:
                age_distribution["3-6 months"] += 1
            elif days_since_access < 365:
                age_distribution["6-12 months"] += 1
            else:
                age_distribution[">1 year"] += 1

            # Categorize
            if days_since_access >= SOFT_DELETE_DAYS:
                if isinstance(importance, (int, float)) and importance >= 80:
                    stats["protected_high_importance"] += 1
                    continue
                stats["soft_delete_candidates"] += 1
                soft_delete_candidates.append(
                    {
                        "id": point.id,
                        "days_old": round(days_since_access, 1),
                        "importance": importance,
                        "text_preview": payload.get("text", "")[:80],
                        "source": payload.get("source", ""),
                    }
                )
            elif days_since_access >= ARCHIVE_DAYS and importance < LOW_IMPORTANCE_THRESHOLD:
                stats["archive_candidates"] += 1
                archive_candidates.append(
                    {
                        "id": point.id,
                        "days_old": round(days_since_access, 1),
                        "importance": importance,
                        "text_preview": payload.get("text", "")[:80],
                        "source": payload.get("source", ""),
                    }
                )
            else:
                stats["active"] += 1

            if limit and stats["total"] >= limit:
                break

        if limit and stats["total"] >= limit:
            break

        offset = next_offset
        if offset is None:
            break

        # Progress
        if stats["total"] % 10000 == 0:
            print(f"  Scanned {stats['total']:,}...")

    return stats, archive_candidates, soft_delete_candidates, age_distribution


def recover_pending_archives(execute=False):
    """Recover points marked pending_archive and complete archive/delete."""
    ensure_archive_collection()
    recovered = 0

    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="pending_archive", match=MatchValue(value=True))]),
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )
    except Exception:
        return 0

    if not points:
        return 0

    if not execute:
        return len(points)

    try:
        archive_points = []
        ids = []
        for p in points:
            payload = dict(p.payload) if p.payload else {}
            payload["archived_at"] = datetime.now().isoformat()
            payload.setdefault("archive_reason", "pending_recovery")
            archive_points.append(PointStruct(id=p.id, vector=cast(Any, p.vector), payload=payload))
            ids.append(p.id)

        qdrant.upsert(collection_name=ARCHIVE_COLLECTION, points=archive_points)
        qdrant.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=ids))
        recovered = len(ids)
    except Exception as e:
        print(f"[ERROR] Pending archive recovery failed: {e}")

    return recovered


def archive_memories(candidates, execute=False):
    """Move memories to archive collection."""
    if not candidates:
        return 0

    ensure_archive_collection()

    archived = 0
    batch_size = 50

    ids = [c["id"] for c in candidates]

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]

        try:
            # Fetch full points with vectors
            points = qdrant.retrieve(
                collection_name=COLLECTION,
                ids=batch_ids,
                with_vectors=True,
                with_payload=True,
            )

            if not points:
                continue

            if execute:
                now_iso = datetime.now().isoformat()
                qdrant.set_payload(
                    collection_name=COLLECTION,
                    points=batch_ids,
                    payload={"pending_archive": True, "archive_started": now_iso},
                )

                archive_points = []
                for p in points:
                    payload = dict(p.payload) if p.payload else {}
                    payload["archived_at"] = now_iso
                    payload["archive_reason"] = "decay_low_importance"
                    payload["pending_archive"] = True
                    archive_points.append(
                        PointStruct(
                            id=p.id,
                            vector=cast(Any, p.vector),
                            payload=payload,
                        )
                    )

                qdrant.upsert(
                    collection_name=ARCHIVE_COLLECTION,
                    points=archive_points,
                )

                qdrant.delete(
                    collection_name=COLLECTION,
                    points_selector=PointIdsList(points=batch_ids),
                )

                archived += len(points)
                print(f"  Archived batch: {len(points)} memories")
            else:
                archived += len(points)
        except Exception as e:
            print(f"  [ERROR] Archive batch failed: {e}")

    return archived


def soft_delete_memories(candidates, execute=False):
    """Mark memories as soft-deleted in archive (set soft_deleted flag)."""
    if not candidates:
        return 0

    ensure_archive_collection()
    soft_deleted = 0
    batch_size = 50

    ids = [c["id"] for c in candidates]

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]

        try:
            # First check if they're already in archive
            # If not, archive them first
            points = qdrant.retrieve(
                collection_name=COLLECTION,
                ids=batch_ids,
                with_vectors=True,
                with_payload=True,
            )

            if execute and points:
                now_iso = datetime.now().isoformat()
                qdrant.set_payload(
                    collection_name=COLLECTION,
                    points=batch_ids,
                    payload={"pending_archive": True, "archive_started": now_iso},
                )

                archive_points = []
                for p in points:
                    payload = dict(p.payload) if p.payload else {}
                    payload["archived_at"] = now_iso
                    payload["archive_reason"] = "decay_soft_delete"
                    payload["soft_deleted"] = True
                    payload["pending_archive"] = True
                    archive_points.append(
                        PointStruct(
                            id=p.id,
                            vector=cast(Any, p.vector),
                            payload=payload,
                        )
                    )

                qdrant.upsert(
                    collection_name=ARCHIVE_COLLECTION,
                    points=archive_points,
                )

                qdrant.delete(
                    collection_name=COLLECTION,
                    points_selector=PointIdsList(points=batch_ids),
                )

                soft_deleted += len(points)
                print(f"  Soft-deleted batch: {len(points)} memories")
            else:
                soft_deleted += len(points) if points else len(batch_ids)
        except Exception as e:
            print(f"  [ERROR] Soft-delete batch failed: {e}")

    return soft_deleted


def run_decay(execute=False, stats_only=False, limit=None):
    """Main decay loop."""
    print(f"{'=' * 60}")
    print("RASPUTIN Memory Decay Engine")
    print(f"{'=' * 60}")
    print(f"Mode: {'🔴 EXECUTE' if execute else '🟢 DRY RUN'}")
    print(f"Archive threshold: {ARCHIVE_DAYS} days + importance < {LOW_IMPORTANCE_THRESHOLD}")
    print(f"Soft-delete threshold: {SOFT_DELETE_DAYS} days")
    print()

    # Check collections
    pending = recover_pending_archives(execute=execute)
    if pending:
        print(f"Recovered pending archives: {pending}")

    try:
        info = qdrant.get_collection(COLLECTION)
        print(f"Collection: {COLLECTION} ({info.points_count:,} points)")
    except Exception as e:
        print(f"[FATAL] Cannot connect to Qdrant: {e}")
        sys.exit(1)

    print("\nScanning memories...")
    t0 = time.time()
    stats, archive_cands, softdel_cands, age_dist = scan_memories(limit=limit)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SCAN RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Total scanned: {stats['total']:,}")
    print(f"Active (healthy): {stats['active']:,}")
    print(f"Archive candidates (>{ARCHIVE_DAYS}d, low importance): {stats['archive_candidates']:,}")
    print(f"Soft-delete candidates (>{SOFT_DELETE_DAYS}d): {stats['soft_delete_candidates']:,}")
    print(f"No date (skipped): {stats['no_date']:,}")

    print("\nAge Distribution:")
    for bucket in ["<1 week", "1-4 weeks", "1-3 months", "3-6 months", "6-12 months", ">1 year"]:
        count = age_dist.get(bucket, 0)
        bar = "█" * min(count // 500, 50)
        print(f"  {bucket:>12s}: {count:>6,} {bar}")

    if stats_only:
        return

    # Show top archive candidates
    if archive_cands:
        print("\nTop archive candidates (lowest importance):")
        top = sorted(archive_cands, key=lambda x: x["importance"])[:5]
        for c in top:
            print(
                f"  ID={c['id']} | {c['days_old']}d old | imp={c['importance']} | "
                f"[{c['source']}] {c['text_preview']}..."
            )

    if softdel_cands:
        print("\nTop soft-delete candidates (oldest):")
        top = sorted(softdel_cands, key=lambda x: x["days_old"], reverse=True)[:5]
        for c in top:
            print(
                f"  ID={c['id']} | {c['days_old']}d old | imp={c['importance']} | "
                f"[{c['source']}] {c['text_preview']}..."
            )

    # Execute
    if execute:
        if archive_cands:
            print(f"\n🔴 Archiving {len(archive_cands)} memories...")
            archived = archive_memories(archive_cands, execute=True)
            print(f"  ✅ Archived: {archived}")

        if softdel_cands:
            print(f"\n🔴 Soft-deleting {len(softdel_cands)} memories...")
            deleted = soft_delete_memories(softdel_cands, execute=True)
            print(f"  ✅ Soft-deleted: {deleted}")
    else:
        total_affected = len(archive_cands) + len(softdel_cands)
        if total_affected:
            print(f"\n🟢 DRY RUN: Would affect {total_affected} memories. Use --execute to proceed.")
        else:
            print("\n✅ All memories are healthy. No action needed.")


if __name__ == "__main__":
    _lock_fd = acquire_lock()
    parser = argparse.ArgumentParser(description="RASPUTIN Memory Decay Engine")
    parser.add_argument("--execute", action="store_true", help="Actually archive/delete (default: dry run)")
    parser.add_argument("--stats", action="store_true", help="Show age distribution only")
    parser.add_argument("--limit", type=int, default=None, help="Max memories to scan")
    args = parser.parse_args()

    run_decay(execute=args.execute, stats_only=args.stats, limit=args.limit)
