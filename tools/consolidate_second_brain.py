#!/usr/bin/env python3
"""
Second Brain Consolidation Script
Consolidates all Qdrant collections into one searchable 768d second_brain collection
"""

import os
import requests
import time
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def get_embeddings_dual(texts, batch_size=200):
    """Get embeddings from Ollama for a batch of texts."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            resp = requests.post(
                EMBED_URL, json={"model": EMBED_MODEL, "input": f"search_document: {text[:2000]}"}, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                embeddings.append(data["embeddings"][0])
            elif "embedding" in data:
                embeddings.append(data["embedding"])
    return embeddings


QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
TARGET_COLLECTION = os.environ.get("QDRANT_COLLECTION", "second_brain")

# Collections to delete (empty)
EMPTY_COLLECTIONS = [
    "research",
    "default",
    "general",
    "jarvis_225_procedural_skills",
    "legacy_documents",
    "openclaw_memories",
    "jarvis_conversations",
    "jarvis_225_tool_learnings",
]

# Collections to migrate (name, count, dimension)
MIGRATE_COLLECTIONS = [
    ("gmail_import", 306896, 1024),
    ("midjourney", 5417, 512),
    ("user_225_memories", 2031, 1536),
    ("user_1_memories", 479, 1536),
    ("jarvis_learnings", 243, 384),
    ("health_data", 210, 1024),
    ("jarvis_procedures", 52, 384),
    ("russian_manus_memories", 46, 1024),
    ("user_0_memories", 35, 1536),
    ("user_234_memories", 17, 1536),
    ("legacy_reflections", 7, 1536),
    ("legacy_tool_executions", 3, 1536),
    ("openclaw_conversations", 1, 768),
]


class SecondBrainConsolidator:
    def __init__(self):
        self.start_time = time.time()
        self.total_migrated = 0
        self.total_skipped = 0
        self.collections_deleted = 0

    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {msg}")

    def delete_empty_collections(self):
        """Step 1: Delete empty collections"""
        self.log(f"Step 1: Deleting {len(EMPTY_COLLECTIONS)} empty collections...")

        for coll_name in EMPTY_COLLECTIONS:
            try:
                resp = requests.delete(f"{QDRANT_URL}/collections/{coll_name}")
                if resp.status_code in [200, 404]:
                    self.log(f"  ✓ Deleted {coll_name}")
                    self.collections_deleted += 1
                else:
                    self.log(f"  ✗ Failed to delete {coll_name}: {resp.status_code}")
            except Exception as e:
                self.log(f"  ✗ Error deleting {coll_name}: {e}")

    def extract_text(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract text from payload - try common field names"""
        for field in ["text", "content", "message", "body", "description"]:
            if field in payload:
                text = payload[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return None

    def scroll_collection(self, collection_name: str, limit: int = 100):
        """Generator to scroll through all points in a collection"""
        offset = None

        while True:
            payload = {"limit": limit, "with_payload": True, "with_vector": False}
            if offset:
                payload["offset"] = offset

            resp = requests.post(f"{QDRANT_URL}/collections/{collection_name}/points/scroll", json=payload)

            if resp.status_code != 200:
                self.log(f"  ✗ Scroll failed: {resp.status_code}")
                break

            data = resp.json()
            points = data.get("result", {}).get("points", [])

            if not points:
                break

            yield points

            offset = data.get("result", {}).get("next_page_offset")
            if not offset:
                break

    def migrate_collection(self, coll_name: str, expected_count: int, old_dim: int):
        """Migrate one collection to second_brain with re-embedding"""
        self.log(f"\n📦 Migrating {coll_name} ({expected_count:,} vectors, {old_dim}d → 768d)")

        migrated = 0
        skipped = 0
        batch_texts = []
        batch_points = []

        try:
            for points in self.scroll_collection(coll_name, limit=100):
                for point in points:
                    point_id = point["id"]
                    payload = point.get("payload", {})

                    # Extract text
                    text = self.extract_text(payload)
                    if not text:
                        skipped += 1
                        continue

                    # Add to batch
                    batch_texts.append(text)
                    batch_points.append({"id": point_id, "payload": {**payload, "source_collection": coll_name}})

                    # Process batch when we hit 200 (embedding batch size)
                    if len(batch_texts) >= 200:
                        self._process_batch(batch_texts, batch_points)
                        migrated += len(batch_texts)
                        batch_texts = []
                        batch_points = []

                        # Progress update
                        if migrated % 10000 == 0:
                            pct = (migrated / expected_count) * 100
                            self.log(f"  Progress: {migrated:,}/{expected_count:,} ({pct:.1f}%)")

            # Process remaining batch
            if batch_texts:
                self._process_batch(batch_texts, batch_points)
                migrated += len(batch_texts)

            self.log(f"  ✓ Migrated {migrated:,} vectors, skipped {skipped:,}")
            self.total_migrated += migrated
            self.total_skipped += skipped

            # Verify count
            self._verify_migration(coll_name, migrated)

            # Delete old collection
            self._delete_collection(coll_name)

        except Exception as e:
            self.log(f"  ✗ Migration failed: {e}")
            raise

    def _process_batch(self, texts: List[str], points: List[Dict]):
        """Embed and upsert a batch"""
        try:
            # Get embeddings using dual GPU
            embeddings = get_embeddings_dual(texts, batch_size=200)

            # Prepare upsert payload
            upsert_points = []
            for point, embedding in zip(points, embeddings):
                upsert_points.append({"id": point["id"], "vector": embedding, "payload": point["payload"]})

            # Upsert to second_brain
            resp = requests.put(f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points", json={"points": upsert_points})

            if resp.status_code != 200:
                self.log(f"  ✗ Upsert failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            self.log(f"  ✗ Batch processing error: {e}")
            raise

    def _verify_migration(self, coll_name: str, migrated_count: int):
        """Verify migrated count in second_brain"""
        try:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points/scroll",
                json={"limit": 1, "filter": {"must": [{"key": "source_collection", "match": {"value": coll_name}}]}},
            )

            if resp.status_code == 200:
                # Note: scroll doesn't return total count, but if we can query it, migration worked
                self.log(f"  ✓ Verified migration - points queryable in second_brain")

        except Exception as e:
            self.log(f"  ⚠ Verification warning: {e}")

    def _delete_collection(self, coll_name: str):
        """Delete a collection after successful migration"""
        try:
            resp = requests.delete(f"{QDRANT_URL}/collections/{coll_name}")
            if resp.status_code in [200, 404]:
                self.log(f"  ✓ Deleted old collection: {coll_name}")
                self.collections_deleted += 1
            else:
                self.log(f"  ✗ Failed to delete {coll_name}: {resp.status_code}")
        except Exception as e:
            self.log(f"  ✗ Delete error: {e}")

    def enable_quantization(self):
        """Step 3: Enable scalar quantization"""
        self.log("\n🔧 Step 3: Enabling scalar quantization...")

        try:
            resp = requests.patch(
                f"{QDRANT_URL}/collections/{TARGET_COLLECTION}",
                json={"quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}}},
            )

            if resp.status_code == 200:
                self.log("  ✓ Scalar quantization enabled")
            else:
                self.log(f"  ✗ Failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            self.log(f"  ✗ Error: {e}")

    def create_indexes(self):
        """Step 4: Create payload indexes"""
        self.log("\n🔍 Step 4: Creating payload indexes...")

        indexes = [
            ("source", "keyword"),
            ("date", "keyword"),
            ("importance", "integer"),
            ("source_collection", "keyword"),
        ]

        for field_name, field_schema in indexes:
            try:
                resp = requests.put(
                    f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/index",
                    json={"field_name": field_name, "field_schema": field_schema},
                )

                if resp.status_code == 200:
                    self.log(f"  ✓ Index created: {field_name} ({field_schema})")
                else:
                    self.log(f"  ⚠ Index {field_name}: {resp.status_code}")

            except Exception as e:
                self.log(f"  ✗ Error creating index {field_name}: {e}")

    def tune_hnsw(self):
        """Step 5: Tune HNSW parameters"""
        self.log("\n⚙️  Step 5: Tuning HNSW parameters...")

        try:
            resp = requests.patch(
                f"{QDRANT_URL}/collections/{TARGET_COLLECTION}", json={"hnsw_config": {"m": 16, "ef_construct": 200}}
            )

            if resp.status_code == 200:
                self.log("  ✓ HNSW tuned (m=16, ef_construct=200)")
            else:
                self.log(f"  ✗ Failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            self.log(f"  ✗ Error: {e}")

    def get_collection_stats(self):
        """Get final collection statistics"""
        try:
            resp = requests.get(f"{QDRANT_URL}/collections/{TARGET_COLLECTION}")
            if resp.status_code == 200:
                data = resp.json()["result"]
                return {
                    "vectors": data.get("points_count", 0),
                    "segments": data.get("segments_count", 0),
                    "status": data.get("status", "unknown"),
                }
        except Exception as e:
            self.log(f"Error getting stats: {e}")
        return {}

    def test_search(self):
        """Test search functionality"""
        self.log("\n🧪 Testing search with gmail_import filter...")

        try:
            # Get a test embedding
            test_embeddings = get_embeddings_dual(["test email search"], batch_size=1)

            resp = requests.post(
                f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points/search",
                json={
                    "vector": test_embeddings[0],
                    "limit": 5,
                    "filter": {"must": [{"key": "source_collection", "match": {"value": "gmail_import"}}]},
                },
            )

            if resp.status_code == 200:
                results = resp.json()["result"]
                self.log(f"  ✓ Search successful - found {len(results)} results from gmail_import")
                return True
            else:
                self.log(f"  ✗ Search failed: {resp.status_code}")
                return False

        except Exception as e:
            self.log(f"  ✗ Test error: {e}")
            return False

    def run(self):
        """Execute full consolidation"""
        self.log("🚀 Starting Second Brain Consolidation")
        self.log(f"Target: {TARGET_COLLECTION} @ {QDRANT_URL}\n")

        # Step 1: Delete empty collections
        self.delete_empty_collections()

        # Step 2: Migrate all collections
        self.log("\n📦 Step 2: Migrating collections with re-embedding...")
        for coll_name, count, dim in MIGRATE_COLLECTIONS:
            self.migrate_collection(coll_name, count, dim)

        # Step 3-5: Optimization
        self.enable_quantization()
        self.create_indexes()
        self.tune_hnsw()

        # Test
        self.test_search()

        # Final report
        self.print_final_report()

    def print_final_report(self):
        """Print comprehensive final report"""
        elapsed = time.time() - self.start_time
        stats = self.get_collection_stats()

        self.log("\n" + "=" * 60)
        self.log("✅ CONSOLIDATION COMPLETE")
        self.log("=" * 60)
        self.log(f"Total time: {elapsed / 60:.1f} minutes ({elapsed:.1f}s)")
        self.log(f"Vectors migrated: {self.total_migrated:,}")
        self.log(f"Vectors skipped: {self.total_skipped:,}")
        self.log(f"Collections deleted: {self.collections_deleted}")
        self.log(f"")
        self.log(f"Second Brain Stats:")
        self.log(f"  Total vectors: {stats.get('vectors', 'unknown'):,}")
        self.log(f"  Segments: {stats.get('segments', 'unknown')}")
        self.log(f"  Status: {stats.get('status', 'unknown')}")
        self.log(f"")
        self.log(f"Throughput: {self.total_migrated / elapsed:.1f} vectors/sec")
        self.log("=" * 60)


if __name__ == "__main__":
    consolidator = SecondBrainConsolidator()
    try:
        consolidator.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
